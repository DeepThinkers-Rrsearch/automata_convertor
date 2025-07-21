import base64
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from prompt_templates.text_extraction import dfa_minimization_extraction_prompt


def extract_dfa_text_from_image(image_bytes: bytes) -> str:
    """
    Takes raw image bytes of a DFA diagram, sends it to Gemini model,
    and extracts the transitions in strict DFA format.

    Returns:
        A string containing the DFA in the expected format.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": dfa_minimization_extraction_prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image}"},
        ]
    )

    response = llm.invoke([message])
    return response.content