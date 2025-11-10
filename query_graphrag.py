#!/usr/bin/env python3
"""
Query the GraphRAG knowledge graph using the CLI interface.

This script provides a wrapper around the graphrag CLI query commands
for easier interaction with the safety incident knowledge graph.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
GRAPHRAG_ROOT = Path("./graphRAG")

def run_query(query_type: str, question: str):
    """
    Run a GraphRAG query using the CLI.

    Args:
        query_type: 'local' or 'global'
        question: The question to ask
    """

    if query_type not in ['local', 'global']:
        print(f"Error: Query type must be 'local' or 'global', not '{query_type}'")
        return

    print(f"\n{'='*60}")
    print(f"{query_type.upper()} SEARCH: {question}")
    print('='*60)
    print()

    # Build the command - use full path to graphrag
    graphrag_path = Path.home() / "miniforge3/envs/graphrag_env/bin/graphrag"
    if not graphrag_path.exists():
        graphrag_path = "graphrag"  # Fall back to PATH

    cmd = [
        str(graphrag_path),
        'query',
        f'--root={GRAPHRAG_ROOT}',
        f'--method={query_type}',
        '--query', question
    ]

    try:
        # Run the query - suppress stderr to hide Ollama noise
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Suppress Ollama logging noise
            text=True,
            check=True
        )

        # Print the output
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error running query: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    except FileNotFoundError:
        print("Error: 'graphrag' command not found.")
        print("Make sure GraphRAG is installed and in your PATH.")
        print("Try: pip install graphrag")

def interactive_mode():
    """Interactive query mode."""

    print("\n" + "="*60)
    print("INTERACTIVE GRAPHRAG QUERY MODE")
    print("="*60)
    print("\nCommands:")
    print("  local <question>  - Local search (specific entities/events)")
    print("  global <question> - Global search (patterns/communities)")
    print("  quit             - Exit")
    print("\nExamples:")
    print("  local What incidents involved stairs?")
    print("  global What are the most common injury types?")
    print("="*60)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Parse command
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Please specify 'local' or 'global' followed by your question.")
                continue

            command, question = parts[0].lower(), parts[1]

            if command in ['local', 'global']:
                run_query(command, question)
            else:
                print(f"Unknown command: {command}. Use 'local' or 'global'.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main entry point."""

    # Check if output directory exists
    if not GRAPHRAG_ROOT.exists():
        print(f"Error: GraphRAG root directory not found: {GRAPHRAG_ROOT}")
        print("Please make sure you've run GraphRAG indexing first.")
        return

    output_dir = GRAPHRAG_ROOT / "output"
    if not output_dir.exists():
        print(f"Error: GraphRAG output directory not found: {output_dir}")
        print("Please run GraphRAG indexing first: sbatch submit.sbatch")
        return

    # Check for command-line query
    if len(sys.argv) > 1:
        # Non-interactive mode: run single query
        query_type = sys.argv[1].lower()
        if len(sys.argv) < 3:
            print("Usage: python query_graphrag.py [local|global] '<question>'")
            print("\nOr run without arguments for interactive mode:")
            print("       python query_graphrag.py")
            return

        question = " ".join(sys.argv[2:])
        run_query(query_type, question)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
