# HW-3_Weekend-movie-trip
EECS 731:Assignment 3: HW#3_Weekend movie trip

Notebook: **Weekend_movie_trip.ipynb** Purpose: Deduced Additional Information, visualization and knn clustering modelling 

1. Loaded the links.csv, movies.csv, ratings.csv, tags.csv in 4 different dataframes.

2. **Additional Information #1**: To find the userId that has rated maximum number of movies.

3. Converted the above new data (from 2) into a new data frame (df_ratings_max_userid).

4. Plotted a graph to show: number of movies rated against userId.

5. **Additional Information #2**: To find the userId that has tagged maximum number of movies.

6. Converted the above new data (from 5) into a dataframe (df_tag_max_userid)

7. Plotted a graph to show: number of movies tagged against userId.

8. **Additional Information #3:** To find the number of movies rated and tagged by each user (df_Merged).

9. **Additional Information #4**: Converted the timestamp from millisecond format to 2000-07-30 18:45:03

10. Created 3 new columns namely year, month and date for each of the userId

11. **Additional Information #5**: To find for each year how many users rated the movies (df_ratings_timeStamp_merged_newYEAR).

12. **Additional Information #6**:To find for each month how many users rated the movies (df_ratings_timeStamp_merged_newMONTH).

**My findings** -->
* userId: 414  rated maximum number of 2698 movies.
* userId: 53 (+13)   rated minimum number of 20 movies.

* userId: 474  tagged maximum number of 1507 movies.
* userId: 288,274,300,543,7,161,600,167  tagged minimum number of 1 movies.

* The userId 414 that rated maximum number of 2698 movies but din't tag any of the movies.
* In year 2000, maximum number of 10061 user rated the movies.
* In year 1998, minimum number of 507 user rated the movies.
* Maximum number of user i.e. 10883 rated movies in the month of May.
* Minimum number of user i.e. 6844 rated movies in the month of December.

 **knn model**.

* Merged the dataset into and handled the missing information
* In order to apply knn, changed the datatype of all (object type) attribute to int
* Splited the data into testing and training data with the help of sklearn.model_selection import train_test_split
* Applied knn model.
* accuracy: 40.66%
