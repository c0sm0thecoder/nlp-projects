package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/reader"
	"github.com/xitongsys/parquet-go/writer"
)

/* =======================
   CONFIG
======================= */

const (
	lang = "az"
)

var authors = []string{
	"Məhəmməd Füzuli",
	"İmadəddin Nəsimi",
	"Şah İsmayıl Xətai",
	"Molla Pənah Vaqif",
	"Qasım bəy Zakir",
	"Seyid Əzim Şirvani",
	"Xurşidbanu Natəvan",
	"Qətran Təbrizi",
	"Xaqani Şirvani",
}

/* =======================
   DATA MODEL
======================= */

type PoemRow struct {
	Author string `parquet:"name=author, type=BYTE_ARRAY, convertedtype=UTF8, encoding=PLAIN_DICTIONARY"`
	Title  string `parquet:"name=title, type=BYTE_ARRAY, convertedtype=UTF8, encoding=PLAIN_DICTIONARY"`
	URL    string `parquet:"name=url, type=BYTE_ARRAY, convertedtype=UTF8, encoding=PLAIN_DICTIONARY"`
	Text   string `parquet:"name=text, type=BYTE_ARRAY, convertedtype=UTF8, encoding=PLAIN_DICTIONARY"`
}

/* HTTP CLIENT
======================= */

var client = &http.Client{
	Timeout: 20 * time.Second,
}

/* =======================
   AUTHOR PAGE PARSER
======================= */

func fetchAuthorPoems(author string) ([]string, error) {
	log.Printf("Fetching author page for: %s", author)

	u := fmt.Sprintf(
		"https://%s.wikisource.org/w/api.php?action=parse&page=%s&prop=links&format=json",
		lang,
		url.QueryEscape("Müəllif:"+author),
	)

	req, _ := http.NewRequest("GET", u, nil)
	req.Header.Set("User-Agent", "GoWikisourceScraper/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var data struct {
		Parse struct {
			Links []struct {
				NS    int    `json:"ns"`
				Title string `json:"*"`
			} `json:"links"`
		} `json:"parse"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}

	var titles []string
	for _, l := range data.Parse.Links {
		if l.NS != 0 {
			continue
		}
		if strings.HasPrefix(l.Title, "Kateqoriya:") ||
			strings.HasPrefix(l.Title, "Şəkil:") ||
			strings.HasPrefix(l.Title, "Author:") {
			continue
		}
		titles = append(titles, l.Title)
	}

	return titles, nil
}

/* =======================
   POEM FETCHER
======================= */

func fetchPoemWikitext(title string) (string, error) {
	u := fmt.Sprintf(
		"https://%s.wikisource.org/w/api.php?action=parse&page=%s&prop=wikitext&format=json",
		lang,
		url.QueryEscape(title),
	)

	var resp *http.Response
	var err error

	// Retry loop for rate limiting or transient errors
	for attempt := 0; attempt < 3; attempt++ {
		req, _ := http.NewRequest("GET", u, nil)
		req.Header.Set("User-Agent", "GoWikisourceScraper/1.0 (contact: bot@example.com)")

		resp, err = client.Do(req)
		if err != nil {
			time.Sleep(time.Second * time.Duration(attempt+1))
			continue
		}

		if resp.StatusCode == 429 {
			resp.Body.Close()
			log.Printf("Rate limited fetching %s. Sleeping...", title)
			time.Sleep(time.Second * 5)
			continue
		}

		if resp.StatusCode != 200 {
			resp.Body.Close()
			return "", fmt.Errorf("HTTP %d", resp.StatusCode)
		}

		break
	}

	if resp == nil {
		return "", fmt.Errorf("failed after retries: %v", err)
	}
	defer resp.Body.Close()

	var data struct {
		Parse struct {
			Wikitext struct {
				Text string `json:"*"`
			} `json:"wikitext"`
		} `json:"parse"`
		Error struct {
			Code string `json:"code"`
			Info string `json:"info"`
		} `json:"error"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		// If JSON fails, it might be HTML error page.
		return "", fmt.Errorf("json decode error: %v", err)
	}

	if data.Error.Code != "" {
		return "", fmt.Errorf("API error: %s - %s", data.Error.Code, data.Error.Info)
	}

	return data.Parse.Wikitext.Text, nil
}

/* =======================
   WIKITEXT CLEANER
======================= */

func cleanWikitext(w string) string {
	reTemplate := regexp.MustCompile(`\{\{[\s\S]*?\}\}`)
	reFile := regexp.MustCompile(`\[\[(Şəkil|File|Image):[^\]]+\]\]`)
	reCat := regexp.MustCompile(`\[\[Kateqoriya:[^\]]+\]\]`)
	reInterwiki := regexp.MustCompile(`\[\[[a-z\-_]+:[^\]]+\]\]`) // Remove interwiki links like [[azb:...]]
	reRef := regexp.MustCompile(`<ref[^>]*>[\s\S]*?<\/ref>|<references\/?>`)
	reHead := regexp.MustCompile(`==+.*?==+`)
	rePoem := regexp.MustCompile(`<\/?poem>`) // Remove <poem> tags
	reBr := regexp.MustCompile(`<br\s*\/?>`)  // Replace <br> with newline

	w = reTemplate.ReplaceAllString(w, "")
	w = reFile.ReplaceAllString(w, "")
	w = reCat.ReplaceAllString(w, "")
	w = reInterwiki.ReplaceAllString(w, "")
	w = reRef.ReplaceAllString(w, "")
	w = reHead.ReplaceAllString(w, "")
	w = rePoem.ReplaceAllString(w, "")
	w = reBr.ReplaceAllString(w, "\n")

	// Simplify remaining links: [[Target|Text]] -> Text or [[Target]] -> Target
	reLink := regexp.MustCompile(`\[\[(?:[^|\]]*\|)?([^\]]+)\]\]`)
	w = reLink.ReplaceAllString(w, "$1")

	lines := strings.Split(w, "\n")
	var cleaned []string
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l != "" {
			cleaned = append(cleaned, l)
		}
	}

	return strings.Join(cleaned, "\n")
}

/* =======================
   WORKER POOL
======================= */

type Job struct {
	Author string
	Title  string
}

func worker(id int, jobs <-chan Job, results chan<- PoemRow, wg *sync.WaitGroup) {
	defer wg.Done()
	log.Printf("Worker %d started", id)

	for job := range jobs {
		log.Printf("Worker %d → fetching: %s (%s)", id, job.Title, job.Author)

		raw, err := fetchPoemWikitext(job.Title)
		if err != nil {
			log.Printf("Worker %d ✗ error fetching %s: %v", id, job.Title, err)
			continue
		}

		clean := cleanWikitext(raw)

		if len(clean) < 300 || strings.Count(clean, "\n") < 10 {
			log.Printf("Worker %d ↷ skipped (not a poem): %s", id, job.Title)
			continue
		}

		log.Printf("Worker %d ✓ accepted: %s", id, job.Title)

		results <- PoemRow{
			Author: job.Author,
			Title:  job.Title,
			URL:    fmt.Sprintf("https://%s.wikisource.org/wiki/%s", lang, url.PathEscape(job.Title)),
			Text:   clean,
		}
	}
}

/* =======================
   PARQUET HELPER
======================= */

func readExistingParquet(path string) ([]PoemRow, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return []PoemRow{}, nil
	}

	fr, err := local.NewLocalFileReader(path)
	if err != nil {
		return nil, err
	}
	defer fr.Close()

	pr, err := reader.NewParquetReader(fr, new(PoemRow), 4)
	if err != nil {
		return nil, err
	}
	defer pr.ReadStop()

	num := int(pr.GetNumRows())
	if num == 0 {
		return []PoemRow{}, nil
	}

	res, err := pr.ReadByNumber(num)
	if err != nil {
		return nil, err
	}

	var rows []PoemRow
	for _, r := range res {
		rows = append(rows, r.(PoemRow))
	}
	return rows, nil
}

/* =======================
   PARQUET WRITER
======================= */

func writeParquet(rows []PoemRow, path string) error {
	log.Printf("Writing %d poems to Parquet: %s", len(rows), path)

	fw, err := local.NewLocalFileWriter(path)
	if err != nil {
		return err
	}
	defer fw.Close()

	pw, err := writer.NewParquetWriter(fw, new(PoemRow), 4)
	if err != nil {
		return err
	}
	defer pw.WriteStop()

	for i, r := range rows {
		pw.Write(r)
		if (i+1)%5 == 0 || i+1 == len(rows) {
			log.Printf("Parquet progress: %d / %d", i+1, len(rows))
		}
	}
	return nil
}

/* =======================
   MAIN
======================= */

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	runtime.GOMAXPROCS(runtime.NumCPU())
	log.Printf("Starting scraper (CPUs=%d)", runtime.NumCPU())

	// 1. Gather all jobs (Author + Title)
	var allJobs []Job
	for _, auth := range authors {
		titles, err := fetchAuthorPoems(auth)
		if err != nil {
			log.Printf("Error fetching author %s: %v", auth, err)
			continue
		}
		for _, t := range titles {
			allJobs = append(allJobs, Job{Author: auth, Title: t})
		}
	}

	if len(allJobs) == 0 {
		log.Println("No works found to process.")
		return
	}

	// 2. Process Jobs
	jobs := make(chan Job, len(allJobs))
	results := make(chan PoemRow, len(allJobs))

	// Reduced concurrency to be polite and avoid 429 errors
	workers := 4

	log.Printf("Launching %d workers for %d jobs", workers, len(allJobs))

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go worker(i+1, jobs, results, &wg)
	}

	for _, j := range allJobs {
		jobs <- j
	}
	close(jobs)

	wg.Wait()
	close(results)

	var newRows []PoemRow
	for r := range results {
		newRows = append(newRows, r)
	}

	log.Printf("Collected %d new poems", len(newRows))

	// 3. Append to existing Parquet file
	const parquetFile = "poems.parquet"

	existingRows, err := readExistingParquet(parquetFile)
	if err != nil {
		log.Printf("Warning: Could not read existing parquet file (might be corrupt or empty): %v", err)
		// Proceeding with just new rows, or should we abort?
		// If read fails, maybe backing up old file is better?
		// For now, let's treat it as empty if error, but user said "old data to persist".
		// If error is not IsNotExist, it's risky.
		if !os.IsNotExist(err) {
			log.Printf("Backing up potentially corrupt file to %s.bak", parquetFile)
			os.Rename(parquetFile, parquetFile+".bak")
		}
		existingRows = []PoemRow{}
	}

	log.Printf("Found %d existing poems", len(existingRows))

	// Optional: Deduplication? User didn't ask, but "append" naively might duplicate.
	// User said "new ones to be added". Usually implies uniqueness check,
	// but strict "append" means add to end. I will just append for now to follow "append" instruction literally.
	// If I re-run scraper for same author, I get duplicates.
	// Smart "append" would avoid duplicates.
	// Let's implement deduplication by Title + Author to be safe/useful.

	uniqueMap := make(map[string]bool)
	var finalRows []PoemRow

	// Keep all existing
	for _, r := range existingRows {
		key := r.Author + "|" + r.Title
		uniqueMap[key] = true
		finalRows = append(finalRows, r)
	}

	// Add new if not exists
	addedCount := 0
	for _, r := range newRows {
		key := r.Author + "|" + r.Title
		if !uniqueMap[key] {
			finalRows = append(finalRows, r)
			uniqueMap[key] = true
			addedCount++
		}
	}

	log.Printf("Merging data: %d existing + %d new (filtered from %d fetched) = %d total",
		len(existingRows), addedCount, len(newRows), len(finalRows))

	if err := writeParquet(finalRows, parquetFile); err != nil {
		log.Fatal(err)
	}

	log.Printf("Done ✅  Output: %s", parquetFile)
}
