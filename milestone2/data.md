# Format of data

The Twitter for Sentiment Analysis (T4SA) dataset http://www.t4sa.it/, was given in the paper "Cross-Media Learning for Image Sentiment Analysis in the Wild",  containing ~3.4M number of tweets. Each tweet has been labeled according to the sentiment polarity of the text (negative = 0, neutral = 1, positive = 2) predicted by tandem LSTM-SVM architecture, obtaining a labeled set of tweets in 3 categories. We merged t4sa_text_sentiment.tsv and raw_tweets_text.csv files based on the unique id, applied regex methods for text preprocessing and selected the class with the highest probability as our final label for the dataset (-1 = negative, 0 = neutral, 1 = positive), leaving only label and text columns in our dataset.

The original merged dataset contains the following 5 fields:

1. TWID: The id of the tweet (e.g. 768097627686604801)

2. text: the text of the tweet (e.g. osh Jenkins is looking forward to TAB Breeders Crown Super Sunday https://t.co/antImqAo4Y https://t.co/ejnA78Sks0)

3. NEG: the probability of the tweet falling in the negative class

4. NEU: the probability of the tweet falling in the neutral class

5. POS: the probability of the tweet falling in the positive class


# Examples of original data

768097627686604801,Josh Jenkins is looking forward to TAB Breeders Crown Super Sunday https://t.co/antImqAo4Y https://t.co/ejnA78Sks0, 0.00808970047903,0.04233099488469999,0.9495793046359999 <br />
768097631864102912,RT @2pmthailfans: [Pic] Nichkhun from krjeong86's IG https://t.co/5gcAcu9by7, 0.014643660775799998,0.9265568133679999,0.0587995258562 <br />
768097640278089729,RT @MianUsmanJaved: Congratulations Pakistan on becoming #No1TestTeam in the world against all odds! #JI_PakZindabadRallies https://t.co/1o…, 0.00493932691338,0.029469361925,0.965591311162 <br />
768097627695042560,"RT @PEPalerts: This September, @YESmag is taking you to Maine Mendoza’s surprise thanksgiving party she threw for her fans! https://t.co/oX…", 0.0063891583217,0.0186627694098,0.9749480722689999 <br />
768096868504969216,#Incredible #India #Atulya #Bharat - Land of Seekers #BeProud 🙏 🇮🇳  :|: Plz RT https://t.co/vpghReZWsa, 0.04939848658419999,0.8613946920569999,0.0892068213586 <br />
768097661237026816,"RT @david_gaibis: Newly painted walls, thanks a million to our custodial painters this summer.  Great job ladies!!!#EC_proud https://t.co/…", 0.00788974022764,0.0351228745836,0.9569873851890001 <br />
768097665418747908,RT @CedricFeschotte: Excited to announce: as of July 2017 Feschotte lab will be relocating to @Cornell MBG https://t.co/dd0FG7BRx3, 0.0116560911854,0.0605474970018,0.927796411813 <br />

# Examples of preprocessed data
0,Fundraiser at The Greene Turtle Football FAMILY Fun URL <br />
1,Fantastic shot of two of our top talents in action tonight for AT_USER AT_USER &amp; AT_USER ⚽️ URL <br />
0,AT_USER All eyes are on ElClasico but here's all the football on SuperSport -&gt; URL URL <br />
1,"happy 80th birthday, grandpa. missing you more and more each day❣ URL" <br />
0,"China July industrial profits rise, buoyed by increased sales, falling costs URL URL" <br />
0,RT AT_USER Art Prints URL BlueWhite ATSocialMedia UKSocialMedia UKSOPRO ATSOPRO UKHashtags UKSmallBIz h… <br />
1,"AT_USER electionnight BGC16 myvote2016 LUCKY FAN ur gonna win Hillary, praying for you URL" <br />
0,RT AT_USER 8 days left until kickoff! Check out our countdown here: URL AT_USER AT_USER URL <br />
0,NEXCOM Digital Signage Player Provides Bus Passengers with Real-Time Schedule Updates URL URL <br />
-1,"AT_USER never lack motivation at work ever again, as you were x URL" <br />
0,Monitor official weather bulletins AT_USER AT_USER AT_USER HurricaneMatthew TeamSVL URL <br />


# Number of items in training/dev/test splits

TRAIN size: 1200000
VAL size: 80000
TEST size: 320000
