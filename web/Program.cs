using DinoDeceptionLens.Web.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor(options =>
{
    options.DisconnectedCircuitMaxRetained = 100;
    options.DisconnectedCircuitRetentionPeriod = TimeSpan.FromHours(1);
    options.JSInteropDefaultCallTimeout = TimeSpan.FromHours(1);
    options.MaxBufferedUnacknowledgedRenderBatches = 100;
}).AddHubOptions(options =>
{
    options.ClientTimeoutInterval = TimeSpan.FromHours(1);
    options.HandshakeTimeout = TimeSpan.FromMinutes(5);
    options.KeepAliveInterval = TimeSpan.FromSeconds(10);
    options.MaximumReceiveMessageSize = 10 * 1024 * 1024; // 10MB
    options.StreamBufferCapacity = 100;
    options.MaximumParallelInvocationsPerClient = 10;
});

// Configure HttpClient for API calls
builder.Services.AddHttpClient<ISearchService, SearchService>(client =>
{
    var apiUrl = builder.Configuration.GetValue<string>("ApiUrl") ?? "http://localhost:8000";
    client.BaseAddress = new Uri(apiUrl);
    client.Timeout = TimeSpan.FromHours(1); // Very long timeout for large collections
});

var app = builder.Build();

// Configure the HTTP request pipeline
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();

app.MapBlazorHub();
app.MapFallbackToPage("/_Host");

app.Run();
