For your M2 deliverables, we’ll ask you to submit your data, plus a markdown file named data.md that describes the format of the data. If your data is very large, then you can submit a sample of the data and give a link to a Google Drive that contains the full data set. You data.md should describe the number of items in each of your training/dev/test splits.

It should give an example of the data, describe the file format of the data, give a link to the full data set (if you’re uploading a sample), and give a description of where you collected the data from.

# Format of data

The data given in https://www.kaggle.com/kazanova/sentiment140 contains 1.6 million tweets extracted using the twitter api. The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment. We preprocessed the data so that target, date and text are left.

It contains the following 6 fields:

1. target: the polarity of the tweet (0 = negative, 4 = positive) -> label: (0 = negative, 1 = positive)

2. ids: The id of the tweet ( 2087)

3. date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

4. flag: The query (lyx). If there is no query, then this value is NO_QUERY.

5. user: the user that tweeted (robotickilldozr)

6. text: the text of the tweet (Lyx is cool)

# Examples of data

"0","1467811193","Mon Apr 06 22:19:57 PDT 2009","NO_QUERY","Karoli","@nationwideclass no, it's not behaving at all. i'm mad. why am i here? because I can't see you all over there. "<br />
"0","1467811372","Mon Apr 06 22:20:00 PDT 2009","NO_QUERY","joy_wolf","@Kwesidei not the whole crew "<br />
"0","1467811795","Mon Apr 06 22:20:05 PDT 2009","NO_QUERY","2Hood4Hollywood","@Tatiana_K nope they didn't have it "<br />
"0","1467812025","Mon Apr 06 22:20:09 PDT 2009","NO_QUERY","mimismo","@twittera que me muera ? "<br />
"0","1467812416","Mon Apr 06 22:20:16 PDT 2009","NO_QUERY","erinx3leannexo","spring break in plain city... it's snowing "<br />
"4","2054062289","Sat Jun 06 06:27:36 PDT 2009","NO_QUERY","LiVonLy4tHeLoRd","celebrate ffx with Kelli today!!!!!! YEA! "<br />
"4","2054062297","Sat Jun 06 06:27:36 PDT 2009","NO_QUERY","Sarah_Jeffreys","Boat broken  luckily surrounded by fit men in leathers! "<br />
"4","2054062396","Sat Jun 06 06:27:37 PDT 2009","NO_QUERY","nwwells","@friendsofED Thanks, I see it "<br />
"4","2054062422","Sat Jun 06 06:27:37 PDT 2009","NO_QUERY","NickiGraves","I'm up, I'm up, and buying tickets in 44 minutes.  Wish me luck "

# After preprocessing, examples of data

label,date,text <br />
0,Fri Jun 19 09:34:11 PDT 2009,World hunger hits one billion people. So Sad URL <br />
0,Sun Jun 07 11:36:40 PDT 2009,AT_USER I can't watch it until next week cos i live in England But i'm still excited for next week haha <br />
1,Thu May 14 02:07:04 PDT 2009,just woke up <br />
1,Sat May 30 07:08:52 PDT 2009,"In the garden with soph, fi, craig, and sarah in the sun with wine! So nice to be outside " <br />
1,Wed Jun 03 02:23:00 PDT 2009,AT_USER can't wait to see your new hairstyle!!! <br />
0,Mon May 04 00:44:39 PDT 2009,"AT_USER I miss you, Mr. Superhero. Come back to Texas, and this time actually talk to me instead of hiding on the bus the whole time " <br />


# Number of items in training/dev/test splits

TRAIN size: 1200000
VAL size: 80000
TEST size: 320000
