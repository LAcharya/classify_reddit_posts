


def get_reddit_posts(subreddit, num_posts):
    '''
    returns a dataframe of reddit posts of length num_posts from the subreddit specified. 
    The function also checks for deleted or removed posts and excludes them from the dataframe returned.
    This function uses the pushshift api: https://github.com/pushshift/api
    '''
    import requests
    import pandas as pd
    
    url = 'https://api.pushshift.io/reddit/search/submission'
    params = params = {
        'subreddit':subreddit,
        'size' : 100,
    }
    
    res = requests.get(url, params)
    
    if res.status_code != 200:
        print(f'Error {res.status_code}')
    else:    
        data = res.json() # get data
        posts = data['data'] # get posts

        # get time of last post
        last_post_time = posts[-1]['created_utc']

        # add to dataframe
        df = pd.DataFrame(posts)
        
        # remove any deleted or removed posts
        df = df[df['selftext'] != ('[removed]' or '[deleted]')]
        

        # check length of df against num_posts
        while len(df) < num_posts:
            # add before parameter to params:
            params = params = {
                'subreddit':subreddit,
                'size' : 100,
                'before': last_post_time
            }

            # pull data
            res = requests.get(url, params)
            data = res.json() # get data
            posts = data['data'] # get posts   
            # update last post time
            last_post_time = posts[-1]['created_utc']

           # create dataframe and remove deleted or removed posts
            df_temp = pd.DataFrame(posts)
            df_temp = df_temp[df_temp['selftext'] != ('[removed]' or '[deleted]')]

            # concatenate to existing dataframe 
            df = pd.concat([df, df_temp])
            
            

        return df
    

    
    
def lemmatize(string):
    '''
    returns a lemmatized version of the input string minus english stopwords from nltk
    '''
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    
    lem = WordNetLemmatizer()
    word_list = [lem.lemmatize(word) for word in string.strip().split(' ') if word not in stopwords.words('english')]
    return ' '.join(word_list) 


def plot_conf_matrix(y_true, y_pred, title, filename, cmap, save=1):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix

    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d',
                cmap=cmap,
                yticklabels=['FrontDesk', 'TechSupport'],
                xticklabels=['FrontDesk', 'TechSupport'])
    plt.title(title);
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    if save==1:
        plt.savefig('../figures/' + filename, dpi=300, bbox_inches = "tight")