package com.example;

import org.apache.spark.sql.*;

import java.io.IOException;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class WineQualityPrediction {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityPrediction")
                .getOrCreate();

        // Load the training data
        Dataset<Row> trainingData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("/home/ec2-user/TrainingDataset.csv");

        
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"feature1", "feature2", "feature3"}) // Replace with actual feature names
                .setOutputCol("features");

        trainingData = assembler.transform(trainingData).select("features", "quality");

        
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("quality")
                .setFeaturesCol("features")
                .setMaxIter(10);

        org.apache.spark.ml.classification.LogisticRegressionModel model = lr.fit(trainingData);

        
        Dataset<Row> validationData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("ValidationDataset.csv");

        Dataset<Row> predictions = model.transform(assembler.transform(validationData));

        
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);
        System.out.println("F1 Score: " + f1Score);

        
        try {
            model.save("wine-quality-model");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
