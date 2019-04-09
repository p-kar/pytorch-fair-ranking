import pdb
import json
import urllib.request

base_url = 'https://www.rottentomatoes.com/api/private/v2.0/browse?maxTomato=100&maxPopcorn=100&services=amazon%3Bhbo_go%3Bitunes%3Bnetflix_iw%3Bvudu%3Bamazon_prime%3Bfandango_now&certified&sortBy=release&type=dvd-streaming-all&page='
movie_page_base_url = 'https://www.rottentomatoes.com/'

f = open('movie_urls.txt', 'w')

page = 1
while True:
    data = None
    with urllib.request.urlopen(base_url + str(page)) as url:
        data = json.loads(url.read().decode())
    if data and data['counts']['count'] > 0:
        movie_relative_urls = [d['url'] for d in data['results']]
        movie_urls = [urllib.parse.urljoin(movie_page_base_url, m_url) for m_url in movie_relative_urls]
        for m_url in movie_urls:
            f.write(m_url + '\n')
        page += 1
    else:
        print(page)
        break

f.close()
