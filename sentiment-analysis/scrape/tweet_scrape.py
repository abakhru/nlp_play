import datetime as dt

import pandas as pd
from halo import Halo
from scrape import DATA_DIR
from twitterscraper import query_tweets

begin_date = dt.date(2020, 6, 20)
end_date = dt.date(2020, 6, 28)

limit = 1000
lang = "english"

# run for #PAP , #wpsg, #ProgressSgParty, #GE2020SG
for i in ['#PAP', '#wpsg', '#ProgressSgParty', '#GE2020SG']:
    spinner = Halo(text=f'Downloading "{i}" related tweets ...', spinner='dots')
    spinner.start()
    tweets = query_tweets(i,
                          begindate=begin_date,
                          enddate=end_date,
                          limit=limit,
                          lang=lang)
    df = pd.DataFrame(t.__dict__ for t in tweets)
    file_name = DATA_DIR.joinpath(f"{i.replace('#', '')}.csv")
    df.to_csv(file_name)
    spinner.succeed(f'{i} Done')
