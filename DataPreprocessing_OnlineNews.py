import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('./data/OnlineNewsPopularity/OnlineNewsPopularity.csv', header=0)
print(df.columns)
df_without_nan = df.dropna()


print(df.info())

cols = [' timedelta', ' n_tokens_title', ' n_tokens_content', ' n_unique_tokens', ' n_non_stop_words', ' n_non_stop_unique_tokens', ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos', ' average_token_length',' num_keywords',' kw_min_min',' kw_max_min',' kw_avg_min',' kw_min_max', ' kw_max_max',' kw_avg_max',' kw_min_avg',' kw_max_avg',' kw_avg_avg',' self_reference_min_shares',' self_reference_max_shares',' self_reference_avg_sharess',' is_weekend',' LDA_00',' LDA_01',' LDA_02',' LDA_03',' LDA_04',' global_subjectivity',' global_sentiment_polarity',' global_rate_positive_words',' global_rate_negative_words',' rate_positive_words',' rate_negative_words',' avg_positive_polarity',' min_positive_polarity',' max_positive_polarity',' avg_negative_polarity',' min_negative_polarity',' max_negative_polarity',' title_subjectivity',' title_sentiment_polarity',' abs_title_subjectivity',' abs_title_sentiment_polarity',' shares']
df = df[cols]
df.to_csv("data_for_rrl/Online_News.data", header = None, index = None)


