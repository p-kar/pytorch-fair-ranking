import re
import json
import scrapy
import urllib.parse
from scrapy import Request
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class MovieSpider(CrawlSpider):
    base_url = 'https://wwww.rottentomatoes.com'
    name = 'rottentomatoes.com'
    allowed_domains = ['rottentomatoes.com']
    page_num_regexp = r"\?page=([0-9]+)"

    def start_requests(self):
        with open('movie_urls.json', 'r') as f:
            start_url = [d['url'] for d in json.load(f)]
        with open('movie_urls.txt', 'r') as f:
            start_url.extend([url.strip() for url in f.readlines()])
        start_url = list(set(start_url))
        self.logger.info('Found {} movies'.format(len(start_url)))
        return [Request(s_url + '/reviews/?page=1', self.parse_item) for s_url in start_url]

    def parse_item(self, response):
        self.logger.info('Movie reviews page: %s', response.url)

        # get critic reviews
        if len(response.selector.css('.review_container')) == 0:
            return None

        for sel in response.selector.css('.review_container'):
            review = {}
            review['review_date'] = sel.css('.review_date::text').get().strip()
            review['review_text'] = sel.css('.the_review::text').get().strip()
            if len(sel.xpath('.//div[contains(@class, "review_icon") and contains(@class, "fresh")]')) == 1:
                review['review_flag'] = 'fresh'
            else:
                review['review_flag'] = 'rotten'
            yield review

        page_num = int(re.findall(self.page_num_regexp, response.url)[0])
        next_critic_url = response.url.split('?')[0] + '?page=' + str(page_num + 1)
        yield Request(next_critic_url, self.parse_item)

