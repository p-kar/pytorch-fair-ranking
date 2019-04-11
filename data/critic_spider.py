import re
import scrapy
import urllib.parse
from scrapy import Request
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class CriticSpider(CrawlSpider):
    base_url = 'https://wwww.rottentomatoes.com'
    name = 'rottentomatoes.com'
    allowed_domains = ['rottentomatoes.com']

    critic_base_url = 'https://www.rottentomatoes.com/critics/authors?letter='
    legacy_critic_base_url = 'https://www.rottentomatoes.com/critics/legacy_authors?letter='

    unique_movie_urls = set([])
    page_num_regexp = r"\?page=([0-9]+)"

    def start_requests(self):
        start_url = [self.critic_base_url + chr(ord('a') + i) for i in range(26)]
        start_url.extend([self.legacy_critic_base_url + chr(ord('a') + i) for i in range(26)])
        return [Request(s_url, self.parse_critic_list_page) for s_url in start_url]

    def parse_critic_list_page(self, response):
        self.logger.info('Critic list page: %s', response.url)

        sel = response.selector.xpath('//table[contains(@class, "table-striped")]')
        for row_sel in sel.xpath('.//tr'):
            critic_url = row_sel.xpath('.//td/p/a/@href').get()
            if critic_url is None:
                continue
            critic_url = critic_url.strip()
            critic_url = urllib.parse.urljoin(self.base_url, critic_url + '/movies?page=1')
            yield Request(critic_url, self.parse_critic_page)

    def parse_critic_page(self, response):
        self.logger.info('Critic page: %s', response.url)

        sel = response.selector.xpath('//tbody[@id="review-table-body"]')
        if len(sel.xpath('.//tr')) == 0:
            return None
        for row_sel in sel.xpath('.//tr'):
            movie_url = row_sel.xpath('.//td[contains(@class, "critic-review-table__title-column")]/a/@href').get()
            if movie_url is None:
                continue
            movie_url = movie_url.strip()
            if movie_url in self.unique_movie_urls:
                continue
            if movie_url.startswith('/tv/'):
                continue
            self.unique_movie_urls.add(movie_url)
            yield {'url': movie_url}

        page_num = int(re.findall(self.page_num_regexp, response.url)[0])
        next_critic_url = response.url.split('?')[0] + '?page=' + str(page_num + 1)
        yield Request(next_critic_url, self.parse_critic_page)

