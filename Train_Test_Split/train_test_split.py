from pyspark.sql.functions import percent_rank
from pyspark.sql import Window

class Create_Dataframe_Train_Test_Split:
    def pyspark_train_test_split(self, df, train_split, test_split):
        df=df.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy("GlucoseDisplayTime")))
        train_df=df.where(f"rank <= {train_split}").drop("rank")
        test_df=df.where(f"rank > {test_split}").drop("rank")

        return train_df, test_df

    def pandas_train_test_split(self, train_split, test_split):
        return None