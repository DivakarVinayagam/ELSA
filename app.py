from rich.console import Console
from rich.table import Table
from transformers import pipeline

console = Console()

console.print("\n[bold yellow]Loading model...[/bold yellow]")
analyzer = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
console.clear()

console.print("[bold green]=== GoEmotions CLI ===[/bold green]")
console.print("Type your text below. Type 'exit' or press Enter on an empty line to quit.\n")

while True:
    text = console.input("[cyan]Text > [/cyan]").strip()

    if not text or text.lower() == "exit":
        console.print("\n[dim]Bye![/dim]")
        break

    # Get predictions and keep top emotions (15%+ confidence)
    predictions = analyzer(text)[0]
    results = [p for p in predictions if p["score"] >= 0.15]

    # Render results in a clean table
    table = Table(title=f'Results for: "{text}"', header_style="bold magenta")
    table.add_column("Emotion", style="cyan")
    table.add_column("Confidence", style="green", justify="right")
    table.add_column("Visual", style="yellow")

    for res in results:
        pct = res["score"]
        bar = "■" * int(pct * 10)
        table.add_row(res["label"].title(), f"{pct:.1%}", bar)

    console.print(table)
    console.print()