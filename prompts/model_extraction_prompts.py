from typing import Dict, Any

# Base prompt for extracting variables and relationships
BASE_EXTRACTION_PROMPT = """You are an AI scientific assistant specialized in extracting variables and relationships from qualitative survey data.

Given the following qualitative survey excerpt, extract:
1. Key variables (with clear, concise descriptions)
2. Causal or conditional relationships between variables
3. Hypotheses (if evident in the text)
4. Any moderating or mediating factors mentioned

Format your response as YAML with the following structure (do not use markdown code blocks or backticks):

Variables:
  - VariableName: Brief description of what this variable represents
  - AnotherVariable: Description...

Relationships:
  - If [condition], then [outcome]
  - [Variable1] affects [Variable2] through [mechanism]
  - When [context], [relationship] occurs

Hypotheses:
  - Hypothesis1: Description of the proposed relationship
  - Hypothesis2: Another proposed relationship

Moderators:
  - ModeratorVariable: How it affects the relationship

Context:
{context}

Excerpt:
{input_text}

Please be precise and only extract what is explicitly mentioned or strongly implied in the text. 

IMPORTANT QUALITY REQUIREMENTS:
1. Variables should be specific and measurable concepts
2. Relationships must specify direction (positive/negative) and mechanism
3. Hypotheses should be testable and specific
4. Avoid generic or overly broad statements
5. Each relationship should clearly state how one variable affects another

Return only valid YAML without any markdown formatting."""

# Enhanced prompt with RAG context
RAG_ENHANCED_PROMPT = """You are an AI scientific assistant specialized in extracting variables and relationships from qualitative survey data.

You have access to relevant context from similar survey responses and research documents. Use this context to enhance your understanding and provide more comprehensive model specifications.

Given the following qualitative survey excerpt and relevant context, extract:
1. Key variables (with clear, concise descriptions)
2. Causal or conditional relationships between variables
3. Hypotheses (if evident in the text)
4. Any moderating or mediating factors mentioned
5. Connections to broader themes or patterns from the context

Format your response as YAML with the following structure (do not use markdown code blocks or backticks):

Variables:
  - VariableName: Brief description of what this variable represents
  - AnotherVariable: Description...

Relationships:
  - If [condition], then [outcome]
  - [Variable1] affects [Variable2] through [mechanism]
  - When [context], [relationship] occurs

Hypotheses:
  - Hypothesis1: Description of the proposed relationship
  - Hypothesis2: Another proposed relationship

Moderators:
  - ModeratorVariable: How it affects the relationship

Themes:
  - Theme1: Connection to broader patterns
  - Theme2: Another thematic connection

Relevant Context:
{context}

Current Excerpt:
{input_text}

Please be precise and only extract what is explicitly mentioned or strongly implied in the text, but use the context to identify broader patterns and connections.

IMPORTANT QUALITY REQUIREMENTS:
1. Variables should be specific and measurable concepts
2. Relationships must specify direction (positive/negative) and mechanism
3. Hypotheses should be testable and specific
4. Avoid generic or overly broad statements
5. Each relationship should clearly state how one variable affects another
6. Use context to identify deeper patterns but stay grounded in the text

Return only valid YAML without any markdown formatting."""

# Prompt for thematic analysis
THEMATIC_ANALYSIS_PROMPT = """You are an AI assistant specialized in thematic analysis of qualitative data.

Analyze the following text excerpts and identify:
1. Recurring themes or patterns
2. Key concepts that appear across multiple responses
3. Potential research questions or hypotheses
4. Variables that could be operationalized for quantitative research

Format your response as YAML:

Themes:
  - ThemeName: Description of the theme and its significance
  - AnotherTheme: Description...

KeyConcepts:
  - Concept1: Definition and examples
  - Concept2: Definition and examples

ResearchQuestions:
  - Question1: Specific research question that could be investigated
  - Question2: Another research question

OperationalizableVariables:
  - Variable1: How this could be measured quantitatively
  - Variable2: How this could be measured quantitatively

Text Excerpts:
{text_excerpts}

Please provide a comprehensive analysis that could guide further research."""

# Prompt for model refinement
MODEL_REFINEMENT_PROMPT = """You are an AI assistant helping to refine and validate scientific model specifications.

Review the following model specification and suggest improvements:

1. Check for logical consistency
2. Identify missing variables or relationships
3. Suggest clearer variable definitions
4. Identify potential confounding factors
5. Recommend additional hypotheses to test

Original Model:
{original_model}

Context from which it was derived:
{context}

Please provide:
1. A refined version of the model
2. A list of suggested improvements
3. Additional variables or relationships to consider
4. Potential research questions to investigate

Format your response as YAML:

RefinedModel:
  Variables:
    - VariableName: Improved description
  Relationships:
    - Refined relationship statement
  Hypotheses:
    - Refined hypothesis

Suggestions:
  - Suggestion1: Description of improvement
  - Suggestion2: Another improvement

AdditionalConsiderations:
  - Consideration1: Additional factor to consider
  - Consideration2: Another consideration"""

def get_prompt_template(prompt_type: str = "base") -> str:
    """Get the appropriate prompt template based on type."""
    templates = {
        "base": BASE_EXTRACTION_PROMPT,
        "rag": RAG_ENHANCED_PROMPT,
        "thematic": THEMATIC_ANALYSIS_PROMPT,
        "refinement": MODEL_REFINEMENT_PROMPT
    }
    return templates.get(prompt_type, BASE_EXTRACTION_PROMPT)

def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with the given parameters."""
    return template.format(**kwargs) 