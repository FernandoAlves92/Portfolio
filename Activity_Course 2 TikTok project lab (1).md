# **TikTok Project**
**This is part of Google Advanced Analytics Professional Certificate course hosted by Coursera. I do not own the data and branding showcased here**

![alt text](https://1000logos.net/wp-content/uploads/2019/06/Tiktok_Logo.png) 

**Background on the TikTok scenario**
At TikTok, our mission is to inspire creativity and bring joy. Our employees lead with curiosity and move at the speed of culture. Combined with our company's flat structure, you'll be given dynamic opportunities to make a real impact on a rapidly expanding company and grow your career.

TikTok users have the ability to submit reports that identify videos and comments that contain user claims. These reports identify content that needs to be reviewed by moderators. The process generates a large number of user reports that are challenging to consider in a timely manner. 

TikTok is working on the development of a predictive model that can determine whether a video contains a claim or offers an opinion. With a successful prediction model, TikTok can reduce the backlog of user reports and prioritize them more efficiently.

**Project background**
TikTok’s data team is in the earliest stages of the claims classification project. The following tasks are needed before the team can begin the data analysis process:

Build a dataframe for the TikTok dataset

Examine data type of each column

Gather descriptive statistics

**Your assignment**
You will build a dataframe for the claims classification data. After the dataframe is complete, you will organize the claims data for the process of exploratory data analysis, and update the team on your progress and insights.

**The goal** is to construct a dataframe in Python, perform a cursory inspection of the provided dataset, and inform TikTok data team members of your findings.
<br/>

<br/>

### **Task 1. Understand the situation**

The first step involves comprehending the business objective, using it as a lens to interpret the data provided by TikTok. In the second step, the dataset is imported and visualized, preparing it for subsequent analysis. Finally, the previous steps are merged to gain an understanding of the dataset's structure, encompassing its columns, rows, and the pertinent data required to fulfill the business objectives.

To facilitate these tasks, I will employ Python, Numpy, and Pandas documentation to review code and provide opportunities for coding reviews.

Additionally, I will maintain close contact with Data and Business leaders, review their emails and shared documents. This will ensure easy reference for routine consultation and to align tasks with their requirements.


### **Task 2a. Imports and data loading**
In this Task, I will import the pandas and numpy libraries to assis with calculations, data structures and statistical analysis.

```python
### YOUR CODE HERE ###

import pandas as pd
import numpy as np
```

Then, I will load the dataset into a dataframe. Creating a dataframe will help you conduct data manipulation, exploratory data analysis (EDA), and statistical activities.


```python
data = pd.read_csv("tiktok_dataset.csv")
```

### **Task 2b. Understand the data - Inspect the data**

 I will inspect summary information about the dataframe by coding:



```python
### YOUR CODE HERE ###

data.head()
```




</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
### YOUR CODE HERE ###

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19382 entries, 0 to 19381
    Data columns (total 12 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   #                         19382 non-null  int64  
     1   claim_status              19084 non-null  object 
     2   video_id                  19382 non-null  int64  
     3   video_duration_sec        19382 non-null  int64  
     4   video_transcription_text  19084 non-null  object 
     5   verified_status           19382 non-null  object 
     6   author_ban_status         19382 non-null  object 
     7   video_view_count          19084 non-null  float64
     8   video_like_count          19084 non-null  float64
     9   video_share_count         19084 non-null  float64
     10  video_download_count      19084 non-null  float64
     11  video_comment_count       19084 non-null  float64
    dtypes: float64(5), int64(3), object(4)
    memory usage: 1.8+ MB



```python
### YOUR CODE HERE ###

data.describe()

```


</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19382.000000</td>
      <td>1.938200e+04</td>
      <td>19382.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9691.500000</td>
      <td>5.627454e+09</td>
      <td>32.421732</td>
      <td>254708.558688</td>
      <td>84304.636030</td>
      <td>16735.248323</td>
      <td>1049.429627</td>
      <td>349.312146</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5595.245794</td>
      <td>2.536440e+09</td>
      <td>16.229967</td>
      <td>322893.280814</td>
      <td>133420.546814</td>
      <td>32036.174350</td>
      <td>2004.299894</td>
      <td>799.638865</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.234959e+09</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4846.250000</td>
      <td>3.430417e+09</td>
      <td>18.000000</td>
      <td>4942.500000</td>
      <td>810.750000</td>
      <td>115.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9691.500000</td>
      <td>5.618664e+09</td>
      <td>32.000000</td>
      <td>9954.500000</td>
      <td>3403.500000</td>
      <td>717.000000</td>
      <td>46.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>14536.750000</td>
      <td>7.843960e+09</td>
      <td>47.000000</td>
      <td>504327.000000</td>
      <td>125020.000000</td>
      <td>18222.000000</td>
      <td>1156.250000</td>
      <td>292.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19382.000000</td>
      <td>9.999873e+09</td>
      <td>60.000000</td>
      <td>999817.000000</td>
      <td>657830.000000</td>
      <td>256130.000000</td>
      <td>14994.000000</td>
      <td>9599.000000</td>
    </tr>
  </tbody>
</table>
</div>



After this initial review, my considerations are as follow:

Initially, each row of the dataset represents a distinct claim pertaining to a specific video on TikTok. The video itself is identified through a unique identifier (UI). In each row, three types of data are present:

Quantitative Data: These columns encompass quantitative attributes of the video, such as duration in seconds, view counts, and more.
Qualitative Data: Qualitative data involves the transcription text of the video, contributing to the analysis of the video's content and possibly identifying unverified claims made within it.
Boolean Data: Boolean values within certain columns classify the video. These values, like author ban status and verified status, offer supplementary insights into the video's status and its creator.
Preliminary analysis reveals that the dataset encompasses 12 columns and 19,382 rows. The data is distributed across integers, floats, and booleans. Several columns exhibit 298 instances of null values, specifically: claim_status, video_transcription_text, video_view_count, video_like_count, video_share_count, video_download_count, and video_comment_count. This could potentially indicate the necessity for additional data sources or a comprehensive data cleaning process.

Regarding dataset statistics, the duration of videos demonstrates a typical trend, peaking at 60 seconds, which aligns with the app's duration limit. In terms of engagement metrics (view count, like count, share count, download count, comment count), values in the upper 75th percentile are notably higher. The gap between the 75th and 50th percentiles ranges from 25 to 35 times. An exception exists with view count, displaying a substantial leap of 50 times from 9,954 views in the 50th percentile to 504,327 views in the 75th percentile.

Outliers are discernible within the data set, encompassing both videos with only 20 views and no corresponding engagement metrics, as well as instances of maximum engagement metrics values. Notably, all engagement metrics except for video count show outlier values at their upper limits. These observations warrant further scrutiny during the data cleaning process and in preparation for subsequent statistical analysis.



### **Task 2c. Understand the data - Investigate the variables**

In this phase, I will begin to investigate the variables more closely to better understand them.

The project proposal objective is to use machine learning to classify videos as either claims or opinions. A good first step towards understanding the data might therefore be examining the `claim_status` variable. I will determine how many videos there are for each different claim status.


```python
### YOUR CODE HERE ###

status = data.groupby('claim_status')
status.count()


```




</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>likes_per_view</th>
      <th>comments_per_view</th>
      <th>shares_per_view</th>
    </tr>
    <tr>
      <th>claim_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>claim</th>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
      <td>9608</td>
    </tr>
    <tr>
      <th>opinion</th>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
      <td>9476</td>
    </tr>
  </tbody>
</table>
</div>



Next, I will examine the engagement trends associated with each different claim status.

I will start by using Boolean masking to filter the data according to claim status, then calculate the mean and median view counts for each claim status.


```python
# What is the average view count of videos with "claim" status?
### YOUR CODE HERE ###

mask = data[data.claim_status == 'claim']

mask.describe()

```




</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9608.000000</td>
      <td>9.608000e+03</td>
      <td>9608.000000</td>
      <td>9608.000000</td>
      <td>9608.000000</td>
      <td>9608.000000</td>
      <td>9608.000000</td>
      <td>9608.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4804.500000</td>
      <td>5.627264e+09</td>
      <td>32.486886</td>
      <td>501029.452748</td>
      <td>166373.331182</td>
      <td>33026.416216</td>
      <td>2070.952227</td>
      <td>691.164863</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2773.735027</td>
      <td>2.543869e+09</td>
      <td>16.172409</td>
      <td>291349.239825</td>
      <td>147623.370888</td>
      <td>38781.676825</td>
      <td>2424.381846</td>
      <td>1017.216834</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.236285e+09</td>
      <td>5.000000</td>
      <td>1049.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2402.750000</td>
      <td>3.400723e+09</td>
      <td>18.000000</td>
      <td>247003.750000</td>
      <td>43436.750000</td>
      <td>5062.250000</td>
      <td>324.750000</td>
      <td>68.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4804.500000</td>
      <td>5.607672e+09</td>
      <td>32.000000</td>
      <td>501555.000000</td>
      <td>123649.000000</td>
      <td>17997.500000</td>
      <td>1139.500000</td>
      <td>286.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7206.250000</td>
      <td>7.834910e+09</td>
      <td>47.000000</td>
      <td>753088.000000</td>
      <td>255715.250000</td>
      <td>47256.000000</td>
      <td>2935.500000</td>
      <td>886.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9608.000000</td>
      <td>9.999873e+09</td>
      <td>60.000000</td>
      <td>999817.000000</td>
      <td>657830.000000</td>
      <td>256130.000000</td>
      <td>14994.000000</td>
      <td>9599.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
### YOUR CODE HERE ###

mask2 = data[data.claim_status == 'opinion']

mask2.describe()
```




</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9476.000000</td>
      <td>9.476000e+03</td>
      <td>9476.000000</td>
      <td>9476.000000</td>
      <td>9476.000000</td>
      <td>9476.000000</td>
      <td>9476.000000</td>
      <td>9476.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14346.500000</td>
      <td>5.622382e+09</td>
      <td>32.359856</td>
      <td>4956.432250</td>
      <td>1092.729844</td>
      <td>217.145631</td>
      <td>13.677290</td>
      <td>2.697446</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2735.629909</td>
      <td>2.530209e+09</td>
      <td>16.281705</td>
      <td>2885.907219</td>
      <td>964.099816</td>
      <td>252.269583</td>
      <td>16.200652</td>
      <td>4.089288</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9609.000000</td>
      <td>1.234959e+09</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11977.750000</td>
      <td>3.448802e+09</td>
      <td>18.000000</td>
      <td>2467.000000</td>
      <td>289.000000</td>
      <td>34.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>14346.500000</td>
      <td>5.611857e+09</td>
      <td>32.000000</td>
      <td>4953.000000</td>
      <td>823.000000</td>
      <td>121.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16715.250000</td>
      <td>7.853243e+09</td>
      <td>47.000000</td>
      <td>7447.250000</td>
      <td>1664.000000</td>
      <td>314.000000</td>
      <td>19.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19084.000000</td>
      <td>9.999835e+09</td>
      <td>60.000000</td>
      <td>9998.000000</td>
      <td>4375.000000</td>
      <td>1674.000000</td>
      <td>101.000000</td>
      <td>32.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now, I will examine trends associated with the ban status of the author.

I will use `groupby()` to calculate how many videos there are for each combination of categories of claim status and author ban status.


```python
### YOUR CODE HERE ###

grouped_byauthor = data.groupby(['claim_status', 'author_ban_status']).count()

print(grouped_byauthor)

```

                                       #  video_id  video_duration_sec  \
    claim_status author_ban_status                                       
    claim        active             6566      6566                6566   
                 banned             1439      1439                1439   
                 under review       1603      1603                1603   
    opinion      active             8817      8817                8817   
                 banned              196       196                 196   
                 under review        463       463                 463   
    
                                    video_transcription_text  verified_status  \
    claim_status author_ban_status                                              
    claim        active                                 6566             6566   
                 banned                                 1439             1439   
                 under review                           1603             1603   
    opinion      active                                 8817             8817   
                 banned                                  196              196   
                 under review                            463              463   
    
                                    video_view_count  video_like_count  \
    claim_status author_ban_status                                       
    claim        active                         6566              6566   
                 banned                         1439              1439   
                 under review                   1603              1603   
    opinion      active                         8817              8817   
                 banned                          196               196   
                 under review                    463               463   
    
                                    video_share_count  video_download_count  \
    claim_status author_ban_status                                            
    claim        active                          6566                  6566   
                 banned                          1439                  1439   
                 under review                    1603                  1603   
    opinion      active                          8817                  8817   
                 banned                           196                   196   
                 under review                     463                   463   
    
                                    video_comment_count  
    claim_status author_ban_status                       
    claim        active                            6566  
                 banned                            1439  
                 under review                      1603  
    opinion      active                            8817  
                 banned                             196  
                 under review                       463  


Then, I will continue investigating engagement levels, now focusing on `author_ban_status`.

I will do this by calculating the median video share count of each author ban status.


```python
### YOUR CODE HERE ###

author_ban = data.groupby('author_ban_status')
```


```python
### YOUR CODE HERE ###

author_ban.median()
```




</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
    <tr>
      <th>author_ban_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>active</th>
      <td>10966.0</td>
      <td>5.624036e+09</td>
      <td>33.0</td>
      <td>8616.0</td>
      <td>2222.0</td>
      <td>437.0</td>
      <td>28.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>banned</th>
      <td>5304.0</td>
      <td>5.563176e+09</td>
      <td>32.0</td>
      <td>448201.0</td>
      <td>105573.0</td>
      <td>14468.0</td>
      <td>892.0</td>
      <td>209.0</td>
    </tr>
    <tr>
      <th>under review</th>
      <td>6175.5</td>
      <td>5.607722e+09</td>
      <td>31.0</td>
      <td>365245.5</td>
      <td>71204.5</td>
      <td>9444.0</td>
      <td>610.5</td>
      <td>136.5</td>
    </tr>
  </tbody>
</table>
</div>



 I will use `groupby()` to group the data by `author_ban_status`, then use `agg()` to get the count, mean, and median of each of the following columns:
* `video_view_count`
* `video_like_count`
* `video_share_count`


```python
### YOUR CODE HERE ###
author_ban = data.groupby('author_ban_status')

aggr_results = author_ban.agg({
    'video_view_count': ['count', 'mean', 'median'],
    'video_like_count': ['count', 'mean', 'median'],
    'video_share_count': ['count', 'mean', 'median']
})

print(aggr_results)
```

                      video_view_count                          video_like_count  \
                                 count           mean    median            count   
    author_ban_status                                                              
    active                       15383  215927.039524    8616.0            15383   
    banned                        1635  445845.439144  448201.0             1635   
    under review                  2066  392204.836399  365245.5             2066   
    
                                               video_share_count                \
                                mean    median             count          mean   
    author_ban_status                                                            
    active              71036.533836    2222.0             15383  14111.466164   
    banned             153017.236697  105573.0              1635  29998.942508   
    under review       128718.050339   71204.5              2066  25774.696999   
    
                                
                        median  
    author_ban_status           
    active               437.0  
    banned             14468.0  
    under review        9444.0  


Now, I will create three new columns to help better understand engagement rates:
* `likes_per_view`: to represent the number of likes divided by the number of views for each video
* `comments_per_view`: to represent the number of comments divided by the number of views for each video
* `shares_per_view`: to represent the number of shares divided by the number of views for each video


```python
### YOUR CODE HERE ###

data['likes_per_view'] = data['video_like_count'] / data['video_view_count']


### YOUR CODE HERE ###

data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']

### YOUR CODE HERE ###

data['shares_per_view'] = data['video_share_count'] / data['video_view_count']

data.head()

```




</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>likes_per_view</th>
      <th>comments_per_view</th>
      <th>shares_per_view</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.056584</td>
      <td>0.000000</td>
      <td>0.000702</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
      <td>0.549096</td>
      <td>0.004855</td>
      <td>0.135111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
      <td>0.108282</td>
      <td>0.000365</td>
      <td>0.003168</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
      <td>0.548459</td>
      <td>0.001335</td>
      <td>0.079569</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
      <td>0.622910</td>
      <td>0.002706</td>
      <td>0.073175</td>
    </tr>
  </tbody>
</table>
</div>



I will use `groupby()` to compile the information in each of the three newly created columns for each combination of categories of claim status and author ban status, then use `agg()` to calculate the count, the mean, and the median of each group.


```python
### YOUR CODE HERE ###

author_ban = data.groupby('author_ban_status')

aggr_results = author_ban.agg({
    'likes_per_view': ['count', 'mean', 'median'],
    'comments_per_view': ['count', 'mean', 'median'],
    'shares_per_view': ['count', 'mean', 'median']
})

print(aggr_results)

```

                      likes_per_view                     comments_per_view  \
                               count      mean    median             count   
    author_ban_status                                                        
    active                     15383  0.266609  0.254227             15383   
    banned                      1635  0.328503  0.325045              1635   
    under review                2066  0.305227  0.290504              2066   
    
                                          shares_per_view                      
                           mean    median           count      mean    median  
    author_ban_status                                                          
    active             0.000891  0.000421           15383  0.053003  0.038429  
    banned             0.001264  0.000658            1635  0.064613  0.048507  
    under review       0.001181  0.000602            2066  0.060969  0.045357  


This is my summary analysis to the Data and Business team at Tiktok:


Through further analysis, I've unearthed several significant trends and correlations that will greatly contribute to the claims and opinions classification project. The data reveals shared statistical characteristics between the two classifications.

Out of 19,084 non-null rows, there's an even 50% split between rows classified as claims and those designated as opinions. However, it's essential to note that null values are excluded from this calculation, as their classification cannot be assumed.

In terms of Opinion videos, a higher count of authors falls under the "active" classification in comparison to Claim videos (8,817 vs. 6,566). Conversely, Claim videos have a notably greater number of items originating from "banned" authors (1,439 vs. 196) and authors "under review" (1,603 vs. 463). This observation may suggest a correlation between videos categorized as claims and those originating from banned or under review authors.

Another significant correlation surfaces regarding engagement rates among authors. When compared to the "active" author status, both "banned" and "under review" classifications exhibit engagement metrics ranging from 21 to 52 times higher.

This pattern is further reinforced by the statistical analysis of each author status. Average view, like, and share counts are 80% to 100% higher for "banned" and "under review" authors. This suggests that videos from banned or under review authors tend to attract more engagement.

However, a notable distinction arises in the median values for the same metrics. Here, the median values can be up to 52 times higher for "banned" and "under review" authors compared to active ones. This implies that outliers within the active classification might be inflating the mean value. Consequently, the median could be a more conservative measure of the difference.

Lastly, an evaluation of likes, comments, and shares per view was conducted to assess the correlation between views and engagement metrics in each classification. The results once again emphasize that "banned" and "under review" authors receive more engagement per view, as follows:

Median Likes per View:
Active: 0.25
Banned: 0.32
Under review: 0.29

Median Comments per View:
Active: 0.0004
Banned: 0.0006
Under review: 0.0006

Median Shares per View:
Active: 0.038
Banned: 0.048
Under review: 0.045



