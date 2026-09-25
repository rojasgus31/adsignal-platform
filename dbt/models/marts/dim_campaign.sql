-- One row per campaign, carrying both human names.
--
-- A dimension rather than two extra columns on `fct_campaign_daily`: that table
-- has tens of thousands of rows and would repeat both strings on every one.
-- This is ~96 rows and joins once.
--
-- left join, not inner: a campaign whose advertiser_id has no matching row in
-- raw_advertisers must still appear here. fct_campaign_daily joins only
-- stg_campaigns, so an inner join here would silently drop that campaign from
-- the dimension while its fact rows kept accumulating — an orphan with a
-- fact row but nothing to render. Dropping a row is worse than labelling it:
-- the dashboard would show fewer campaigns than its own counts claim, with no
-- indication why.
select
    c.campaign_id,
    c.campaign_name,
    c.advertiser_id,
    coalesce(a.advertiser_name, 'Unknown advertiser') as advertiser_name,
    c.channel,
    c.start_date,
    c.end_date
from {{ ref('stg_campaigns') }} c
left join {{ ref('stg_advertisers') }} a on a.advertiser_id = c.advertiser_id
