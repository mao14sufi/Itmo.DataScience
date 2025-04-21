// See https://aka.ms/new-console-template for more information
using SpamDetector;
Console.WriteLine("Write a message");


//Load sample data
var sampleData = new MLModel1.ModelInput()
{
	Subject =Console.ReadLine(),
};

//Load model and predict output
var result = MLModel1.Predict(sampleData);

Console.WriteLine($"Content - {result.Subject} - THIS IS  {(Convert.ToBoolean(result.IsSpam) ? "SPAM" : "NO SPAM")}. ");