using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace BikeSharingPrediction
{
	internal class Program
	{
		// Путь к файлам данных
		private static string dataPath = "C:\\Users\\tommy\\source\\repos\\BikeSharingPrediction\\Data\\bike_sharing.csv";
		// Классы для обработки данных
		public class BikeRentalData
		{
			[LoadColumn(0)]
			public float Season { get; set; }
			[LoadColumn(1)]
			public float Month { get; set; }
			[LoadColumn(2)]
			public float Hour { get; set; }
			[LoadColumn(3)]
			public float Holiday { get; set; }
			[LoadColumn(4)]
			public float Weekday { get; set; }
			[LoadColumn(5)]
			public float WorkingDay { get; set; }
			[LoadColumn(6)]
			public float WeatherCondition { get; set; }
			[LoadColumn(7)]
			public float Temperature { get; set; }
			[LoadColumn(8)]
			public float Humidity { get; set; }
			[LoadColumn(9)]
			public float Windspeed { get; set; }
			
			[LoadColumn(10)]
			//[ColumnName("Label")]
			public bool RentalType { get; set; }
			     // 0 = краткосрочная, 1 = догосрочная
		}

		public class RentalTypePrediction
		{
			[ColumnName("PredictedLabel")]
			public bool PredictedRentalType { get; set; }
			public float Probability { get; set; }
			public float Score { get; set; }
		}
		static void Main(string[] args)
		{
			Console.OutputEncoding = Encoding.UTF8;
			var mlContext = new MLContext(seed: 0);
			var data = mlContext.Data.LoadFromTextFile<BikeRentalData>(
				path: dataPath,
				hasHeader: true,
				separatorChar: ',',
				allowQuoting: true,
				trimWhitespace: true);
			Console.WriteLine("данные загружены");
			var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
			var pipeline = mlContext.Transforms.CopyColumns("Label", "RentalType")
							.Append(mlContext.Transforms.Categorical.OneHotEncoding("SeasonEncoded", "Season"))
							.Append(mlContext.Transforms.Categorical.OneHotEncoding("HolidayEncoded", "Holiday"))
							.Append(mlContext.Transforms.Categorical.OneHotEncoding("WorkingDayEncoded", "WorkingDay"))
							.Append(mlContext.Transforms.Categorical.OneHotEncoding("WeatherConditionEncoded", "WeatherCondition"))
							.Append(mlContext.Transforms.NormalizeMinMax("Temperature"))
							.Append(mlContext.Transforms.NormalizeMinMax("Humidity"))
							.Append(mlContext.Transforms.NormalizeMinMax("Windspeed"))
							.Append(mlContext.Transforms.Concatenate("Features", "SeasonEncoded", "Month", "Hour", "HolidayEncoded", "Weekday",
								"WorkingDayEncoded", "WeatherConditionEncoded", "Temperature", "Humidity", "Windspeed"));

			Console.WriteLine("FastTree");
			var pipelineOne = pipeline.Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName:"Label",featureColumnName:"Features"));
			Console.WriteLine("начало обучения");
			var modelOne = pipelineOne.Fit(trainTestData.TrainSet);
			Console.WriteLine("конец обучения");
			var predictionEngine = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(modelOne);
			var predictions = modelOne.Transform(trainTestData.TestSet);
			var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label" );
			Console.WriteLine($"Accuracy: {metrics.Accuracy}");
			Console.WriteLine($"AreyUnderRocCurve: {metrics.AreaUnderRocCurve}");
			Console.WriteLine($"F1Score: {metrics.F1Score}");
			var example1 = new BikeRentalData
			{
				Season = 1,
				Month = 1,
				Hour = 1,
				Holiday = 1,
				Weekday = 1,
				WorkingDay = 0,
				WeatherCondition = 1,
				Temperature = 1,
				Humidity = 1,
				Windspeed = 1
			};
			var result = predictionEngine.Predict(example1);
			Console.WriteLine($"Result : {result.PredictedRentalType} with probability : {result.Probability}.");


			//
			Console.WriteLine("LightGbm");
			var pipelineTwo = pipeline.Append(mlContext.BinaryClassification.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "Features"));
			Console.WriteLine("начало обучения");
			var modelTwo = pipelineTwo.Fit(trainTestData.TrainSet);
			Console.WriteLine("конец обучения");
			var predictionEngineTwo = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(modelTwo);
			var predictionsTwo = modelTwo.Transform(trainTestData.TestSet);
			var metricsTwo = mlContext.BinaryClassification.Evaluate(predictionsTwo, "Label");
			Console.WriteLine($"Accuracy: {metricsTwo.Accuracy}");
			Console.WriteLine($"AreyUnderRocCurve: {metricsTwo.AreaUnderRocCurve}");
			Console.WriteLine($"F1Score: {metricsTwo.F1Score}");
			var example2 = new BikeRentalData
			{
				Season = 3,
				Month = 5,
				Hour = 15,
				Holiday = 1,
				Weekday = 7,
				WorkingDay = 0,
				WeatherCondition = 1,
				Temperature = 20,
				Humidity = 10,
				Windspeed = 5
			};
			var result2 = predictionEngineTwo.Predict(example2);
			Console.WriteLine($"Result : {result2.PredictedRentalType} with probability : {result2.Probability}.");









		}
	}
}
