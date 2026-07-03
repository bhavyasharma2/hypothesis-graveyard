import sys
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.capture_agent import capture_hypothesis
from agents.archaeology_agent import analyze_history
from agents.steelman_agent import steelman_hypothesis
from agents.verdict_agent import evaluate_verdict

load_dotenv()
console = Console()

def run_pipeline(raw_text: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        console.print("[bold red]ERROR:[/bold red] GEMINI_API_KEY is not set in the .env file.")
        console.print("Please add a valid Gemini API key to proceed.")
        sys.exit(1)
        
    console.print(Panel("[bold green]Hypothesis Graveyard - Multi-Agent Analysis Pipeline[/bold green]", expand=False))
    
    # 1. Capture Agent
    with console.status("[bold blue]1. Capture Agent: Parsing and structuring hypothesis...[/bold blue]"):
        try:
            captured_hyp = capture_hypothesis(raw_text)
        except Exception as e:
            console.print(f"[bold red]Capture Agent Failed:[/bold red] {e}")
            return
            
    console.print("\n[bold cyan]=== CAPTURE AGENT OUTPUT ===[/bold cyan]")
    console.print(f"[bold]Title:[/bold] {captured_hyp.title}")
    console.print(f"[bold]Domain:[/bold] {captured_hyp.domain}")
    console.print(f"[bold]Core Idea:[/bold] {captured_hyp.core_idea}")
    console.print(f"[bold]Rationale:[/bold] {captured_hyp.rationale}")
    console.print(f"[bold]Assumptions:[/bold]")
    for ass in captured_hyp.assumptions:
        console.print(f"  - {ass}")
    console.print(f"[bold]Proposed Test:[/bold] {captured_hyp.proposed_test}")
    
    # 2. Archaeology Agent
    with console.status("[bold blue]2. Archaeology Agent: Searching graveyard and analyzing history...[/bold blue]"):
        try:
            archaeology_rep = analyze_history(captured_hyp)
        except Exception as e:
            console.print(f"[bold red]Archaeology Agent Failed:[/bold red] {e}")
            return
            
    console.print("\n[bold cyan]=== ARCHAEOLOGY AGENT OUTPUT ===[/bold cyan]")
    console.print(Panel(archaeology_rep.historical_precedents_summary, title="Historical Precedents Summary"))
    
    if archaeology_rep.similar_cases_found:
        table = Table(title="Similar Historical Cases Found")
        table.add_column("Past Hypothesis", style="magenta")
        table.add_column("Outcome", style="yellow")
        table.add_column("Conviction", style="green")
        table.add_column("Lessons Learned", style="white")
        
        for case in archaeology_rep.similar_cases_found:
            table.add_row(
                case.past_hypothesis_text[:60] + "...",
                case.outcome,
                f"{case.conviction_score:.1f}",
                case.lessons_learned[:60] + "..."
            )
        console.print(table)
    else:
        console.print("[yellow]No similar historical cases found in the graveyard.[/yellow]")
        
    console.print(f"[bold]Historical Recommendation:[/bold] {archaeology_rep.recommendation_from_history}")
    
    # 3. Steelman Agent
    with console.status("[bold blue]3. Steelman Agent: Optimizing and strengthening hypothesis...[/bold blue]"):
        try:
            steelman_rep = steelman_hypothesis(captured_hyp, archaeology_rep)
        except Exception as e:
            console.print(f"[bold red]Steelman Agent Failed:[/bold red] {e}")
            return
            
    console.print("\n[bold cyan]=== STEELMAN AGENT OUTPUT ===[/bold cyan]")
    console.print(Panel(f"[bold green]Steelmanned Statement:[/bold green]\n{steelman_rep.steelmanned_statement}", title="Steelmanned Hypothesis"))
    
    console.print("[bold]Key Improvements over Original:[/bold]")
    for imp in steelman_rep.key_improvements:
        console.print(f"  - {imp}")
        
    console.print(Panel(steelman_rep.robust_testing_protocol, title="Robust Testing Protocol"))
    
    # 4. Verdict Agent
    with console.status("[bold blue]4. Verdict Agent: Synthesizing final evaluation...[/bold blue]"):
        try:
            verdict_rep = evaluate_verdict(captured_hyp, archaeology_rep, steelman_rep)
        except Exception as e:
            console.print(f"[bold red]Verdict Agent Failed:[/bold red] {e}")
            return
            
    console.print("\n[bold cyan]=== VERDICT AGENT OUTPUT ===[/bold cyan]")
    
    # Determine color based on conviction score
    score_color = "red"
    if verdict_rep.conviction_score >= 70:
        score_color = "green"
    elif verdict_rep.conviction_score >= 40:
        score_color = "yellow"
        
    console.print(Panel(
        f"[bold]Verdict:[/bold] {verdict_rep.verdict_category}\n"
        f"[bold]Conviction Score:[/bold] [{score_color}]{verdict_rep.conviction_score:.1f}/100[/{score_color}]\n\n"
        f"[bold]Executive Summary:[/bold]\n{verdict_rep.executive_summary}",
        title="Final Evaluation",
        border_style=score_color
    ))
    
    console.print("[bold green]Pros / Strengths:[/bold green]")
    for pro in verdict_rep.pros:
        console.print(f"  [green]+[/green] {pro}")
        
    console.print("[bold red]Cons / Risks:[/bold red]")
    for con in verdict_rep.cons:
        console.print(f"  [red]-[/red] {con}")
        
    console.print("\n[bold]Critical Validation Milestones:[/bold]")
    for ms in verdict_rep.critical_milestones:
        console.print(f"  [cyan]*[/cyan] {ms}")

def main():
    if len(sys.argv) > 1:
        raw_text = " ".join(sys.argv[1:])
    else:
        console.print("[yellow]No hypothesis provided as argument. Please enter it below:[/yellow]")
        raw_text = console.input("[bold blue]Enter Hypothesis: [/bold blue]")
        if not raw_text.strip():
            console.print("[red]Empty hypothesis. Exiting.[/red]")
            sys.exit(1)
            
    run_pipeline(raw_text)

if __name__ == "__main__":
    main()
