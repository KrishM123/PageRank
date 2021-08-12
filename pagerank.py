import os
import random
import re
import sys
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    values = dict()

    if corpus[page]:
        for ele in corpus:
            values[ele] = (1 - damping_factor) / len(corpus)
            if ele in corpus[page]:
                values[ele] += damping_factor / len(corpus[page])
    else:
        for ele in corpus:
            values[ele] = 1 / len(corpus)

    return values


def sample_pagerank(corpus, damping_factor, n):
    pagerank = dict()
    random.seed()
    used = True

    for page in corpus:
        pagerank[page] = 0

    for ele in range(n):
        if used is not None:
            used = random.choices(list(corpus.keys()), k=1)[0]
        else:
            model = transition_model(corpus, used, damping_factor)
            population, weights = zip(*model.items())
            used = random.choices(population, weights=weights, k=1)[0]

        pagerank[used] += 1

    for page in corpus:
        pagerank[page] /= n

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    pagerank = dict()
    new = dict()

    for page in corpus:
        pagerank[page] = 1 / len(corpus)

    repeat = True

    while repeat:
        for page in pagerank:
            total = float(0)

            for page2 in corpus:
                if page in corpus[page2]:
                    total += pagerank[page2] / len(corpus[page2])
                if not corpus[page2]:
                    total += pagerank[page2] / len(corpus)

            new[page] = (1 - damping_factor) / len(corpus) + damping_factor * total

        repeat = False

        for page in pagerank:
            if not math.isclose(new[page], pagerank[page], abs_tol=0.001):
                repeat = True
            pagerank[page] = new[page]

    return pagerank



if __name__ == "__main__":
    main()
