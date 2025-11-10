#!/bin/bash
# Test queries for GraphRAG safety incident knowledge graph

echo "======================================"
echo "GraphRAG Query Tests"
echo "======================================"
echo ""
echo "Note: Make sure Ollama is running in another terminal:"
echo "      ~/bin/ollama serve"
echo ""
echo "Running test queries..."
echo "======================================"

# Test 1: Local search - specific entity
echo ""
echo "Test 1: Local Search - What incidents involved forklifts?"
python query_graphrag.py local "What incidents involved forklifts?"

# Test 2: Global search - patterns
echo ""
echo "Test 2: Global Search - What are the most common incident types?"
python query_graphrag.py global "What are the most common incident types?"

# Test 3: Local search - location specific
echo ""
echo "Test 3: Local Search - What happened at Nusajaya Campus?"
python query_graphrag.py local "What happened at Nusajaya Campus?"

echo ""
echo "======================================"
echo "Done!"
echo ""
echo "To run more queries interactively:"
echo "  python query_graphrag.py"
