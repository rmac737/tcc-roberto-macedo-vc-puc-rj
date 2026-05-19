using ICAONet;

while (true)
{
    var watch = System.Diagnostics.Stopwatch.StartNew();

    ICAONet.ICAONet iCAONet = new ICAONet.ICAONet();
    ICAONetParams parametros = new ICAONetParams()
    {
        DpiResultImage = 500,
        HeightResultImage = 640,
        WidthResultImage = 480,
        RemoveBackground = true,
        Image = System.IO.File.ReadAllBytes(@"E:\temp\testes_amazon_rekognition\roberto_04.jpg")
    };
    var dto = await iCAONet.Evaluate(parametros);

    watch.Stop();
    var elapsedMs = watch.ElapsedMilliseconds;

    Console.WriteLine($"Tempo de Processamento: {elapsedMs} milesegundos.");
}
