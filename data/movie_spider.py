import scrapy
import urllib.parse
from scrapy import Request
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class MovieSpider(CrawlSpider):
    base_url = 'https://wwww.rottentomatoes.com'
    name = 'rottentomatoes.com'
    allowed_domains = ['rottentomatoes.com']
    with open('movie_urls.txt', 'r') as f:
        start_url = [url.strip() for url in f.readlines()]

    def start_requests(self):
        return [Request(s_url, self.parse_item) for s_url in self.start_url]

    def parse_item(self, response):
        self.logger.info('Movie page: %s', response.url)
        item = {}

        # Movie URL
        item['url'] = response.url

        # Movie Name
        item['movie_name'] = response.selector.xpath('//h1[@class="mop-ratings-wrap__title mop-ratings-wrap__title--top"]/text()').get().strip()

        # Tomatometer/Audience Score
        sel = response.selector.css('.mop-ratings-wrap__score')
        tomatometer = None
        audience_score = None
        if len(sel) > 0:
            tomatometer = sel[0].xpath('.//span[contains(@class, "mop-ratings-wrap__percentage")]/text()').get().strip()
        if len(sel) > 1:
            audience_score = sel[1].xpath('.//span[contains(@class, "mop-ratings-wrap__percentage")]/text()').get().strip()
        if not tomatometer:
            tomatometer = 'unknown'
        if not audience_score:
            audience_score = 'unknown'
        item['tomatometer'] = tomatometer
        item['audience_score'] = audience_score

        # Critics/Audience Rating
        sel = response.selector.css('.mop-ratings-wrap__row')
        if len(sel.xpath('.//span[contains(@class, "mop-ratings-wrap__icon") and contains(@class, "certified_fresh")]')):
            critic_rating = 'certified_fresh'
        elif len(sel.xpath('.//span[contains(@class, "mop-ratings-wrap__icon") and contains(@class, "fresh")]')):
            critic_rating = 'fresh'
        elif len(sel.xpath('.//span[contains(@class, "mop-ratings-wrap__icon") and contains(@class, "rotten")]')):
            critic_rating = 'rotten'
        else:
            critic_rating = 'unknown'

        if len(sel.xpath('.//span[contains(@class, "mop-ratings-wrap__icon") and contains(@class, "upright")]')):
            audience_rating = 'upright'
        elif len(sel.xpath('.//span[contains(@class, "mop-ratings-wrap__icon") and contains(@class, "spilled")]')):
            audience_rating = 'spilled'
        else:
            audience_rating = 'unknown'
        item['critic_rating'] = critic_rating
        item['audience_rating'] = audience_rating

        # Critic/Audience Reviews Counted (might have commas in the count)
        sel = response.selector.css('.mop-ratings-wrap__review-totals')
        critic_review_count = None
        if len(sel) > 0:
            critic_review_count = sel[0].xpath('.//small[@class="mop-ratings-wrap__text--small"]/text()').get().strip()
        if not critic_review_count or critic_review_count == 'N/A':
            critic_review_count = 'unknown'

        audience_review_count = None
        if len(sel) > 1:
            audience_review_count = sel[1].xpath('.//small[@class="mop-ratings-wrap__text--small"]/text()').get().strip()
        if not audience_review_count or audience_review_count == 'N/A':
            audience_review_count = 'unknown'
        item['critic_review_count'] = critic_review_count
        item['audience_review_count'] = audience_review_count

        # Movie Metadata
        movie_metadata = {}
        sel = response.selector.css('.content-meta')
        for row_sel in sel.css('.meta-row'):
            label = row_sel.css('.meta-label::text').get().strip()
            if len(row_sel.xpath('.//div[@class="meta-value"]/a')) > 0:
                value = [s.get().strip() for s in row_sel.xpath('.//div[@class="meta-value"]/a/text()')]
            elif len(row_sel.xpath('.//div[@class="meta-value"]/time')) > 0:
                value = row_sel.xpath('.//div[@class="meta-value"]/time/text()')[0].get().strip()
            else:
                value = row_sel.css('.meta-value::text').get().strip()
            movie_metadata[label] = value
        item['movie_metadata'] = movie_metadata

        # Cast
        sel = response.selector.css('.castSection')
        cast = []
        for cast_sel in sel.css('.cast-item'):
            cast_page_url = cast_sel.xpath('.//div[@class="media-body"]/a/@href').get().strip()
            cast_page_url = urllib.parse.urljoin(self.base_url, cast_page_url)
            cast_name = cast_sel.xpath('.//div[@class="media-body"]/a/span/text()').get().strip()
            cast.append({'cast_name': cast_name, 'cast_page_url': cast_page_url})
        item['cast'] = cast

        # Top critics URL
        critics_url = response.selector.xpath('//a[@class="criticHeadersLink small unstyled subtle articleLink" and contains(@href, "top_critics")]/@href').get().strip()
        if critics_url is None:
            item['reviews'] = []
            yield item
        else:
            critics_url = urllib.parse.urljoin(self.base_url, critics_url.strip())
            request = Request(url=critics_url, callback=self.parse_critics)
            request.meta['item'] = item
            yield request

    def parse_critics(self, response):
        self.logger.info('Movie critics page: %s', response.url)
        item = {}
        for k, v in response.meta['item'].items():
            item[k] = v

        reviews = []
        # get critic reviews
        for sel in response.selector.css('.review_container'):
            review = {}
            review['review_date'] = sel.css('.review_date::text').get().strip()
            review['review_text'] = sel.css('.the_review::text').get().strip()
            if len(sel.xpath('.//div[contains(@class, "review_icon") and contains(@class, "fresh")]')) == 1:
                review['review_flag'] = 'fresh'
            else:
                review['review_flag'] = 'rotten'
            reviews.append(review)
        item['reviews'] = reviews

        yield item

