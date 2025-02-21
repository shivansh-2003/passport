import streamlit as st
import json
import base64  
from PIL import Image
from openai import OpenAI  
import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool

# --- Fetch OpenAI API Key Securely ---
openai_api_key = st.secrets["openai"]["api_key"]

# Initialize OpenAI Client
client = OpenAI(api_key=openai_api_key)

# Initialize LLM
llm = ChatOpenAI(temperature=0.3, model_name="gpt-4")

# --- Passport Extraction Function ---
def extract_passport_details(image_path):
    """Extracts passport details from an image using OpenAI's GPT-4 Turbo Vision model."""

    # Encode image in Base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # OpenAI API Call
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract passport details from this image and return the result as JSON with:\n"
                            "- full_name\n"
                            "- passport_number\n"
                            "- nationality\n"
                            "- date_of_birth\n"
                            "- date_of_issue\n"
                            "- date_of_expiry\n"
                            "- issuing_country\n"
                            "If any detail is missing, return 'Not readable'."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=300
    )

    # Parse JSON response and return dictionary
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "Could not parse response into a dictionary"}

# --- Streamlit UI ---
st.title("Passport Data Analyzer")

input_option = st.radio("Choose Input Method:", ("Upload Passport Image", "Enter City/State/District"))

if input_option == "Upload Passport Image":
    uploaded_file = st.file_uploader("Upload Passport Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Passport Image", use_column_width=True)

        temp_image_path = "temp_passport_image.jpg"
        image.save(temp_image_path)

        passport_details = extract_passport_details(temp_image_path)

        st.subheader("Extracted Passport Details:")
        st.write(passport_details)
        os.remove(temp_image_path)

elif input_option == "Enter City/State/District":
    city_input = st.text_input("Enter City (Optional):")
    state_input = st.text_input("Enter State (Optional):")
    district_input = st.text_input("Enter District (Optional):")

    if city_input or state_input or district_input:
        location_string = f"{city_input or ''}, {state_input or ''}, {district_input or ''}"
        st.subheader("Country:")
        st.write(location_string)
if __name__ == "__main__":
    st.run()        
        
