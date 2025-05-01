# Using streamlit and google generativeai sdk to build an agentic image sovler

import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
import json
import typing_extensions as typing


# initiatize with api key from env
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


# define the schema for the solution plan
class Solution_plan(typing.TypedDict):
    """ Schema for the solution plan """
    solution: object
    steps: list[str]

def render_app():
    # Initialize the streamlit app
    st.title("Agentic Image Solver")
    render_upload_image()
    render_plan()
    # render_columns()

def render_columns():
    if not has_session_solplan():
        return

    # render the columns  
    col_solver, col_verifier = st.columns(2)

    with col_solver:
        st.header("Solver")

    with col_verifier:
        st.header("Verifier")
        
    return (col_solver, col_verifier)


def render_upload_image():
    file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if file is not None:
        image = Image.open(file)
        st.session_state.uploaded_image = image
        st.image(image, caption="Uploaded Image", use_column_width=True)


def render_plan():
    """ Render the  plan for solving the image """
    
    # exit if there is no image yet
    if not has_session_image():
        return
    
    st.header("Solution plan")
    response = get_solution_plan()
    
    # save response in session state
    if response is not None:
        st.session_state.solution_plan = response
        # display the markdown data in solution_plan
        st.write(response)


def get_solution_plan():
    """ Call the Gemini API with the upload image and a prompt asking it for
    steps to solve the challenge in the image without actually solving it.
    """

    if not has_session_image():
      return
    
    # Use the uploaded image as the input to the Gemini API
    input_image = st.session_state.uploaded_image

    # define the prompt
    prompt = """Your job is to identify the text in is image.
        How should one determine what is written in this text?
        Please provide a step-by-step solution to this.
        Do not try to solve the challenge itself, but only output the steps.
        Output a json containing the steps in the specified format which defines
        the steps to be performed, and a way to verify that the step was successful.
        {
            "steps": {
                "json": {
                    "1": {
                        "step": "step description",
                        "instructions": "detailed instructions to an LLM for the step",
                        "verficiation": "detailed verification instructions to an LLM for the step",
                    },
                    "2": {
                        "step": "step description",
                        "instructions": "detailed instructions to an LLM for the step",
                        "verficiation": "detailed verification instructions to an LLM for the step",                        
                    },
                    ...
                }
            }
        }
        """
    
    # create a generation config
    generation_config = {
        "response_mime_type": "application/json"
    }
    
    # Call the Gemini API with the uploaded image and a prompt
    model = genai.GenerativeModel("gemini-1.5-flash-001")
    response = model.generate_content([input_image, prompt], 
                           generation_config=generation_config)
    
    return json.loads(response.text)


def has_session_image():
  """ Check if there is an image in the session state """
  return "uploaded_image" in st.session_state \
    and st.session_state.uploaded_image is not None


def has_session_solplan():
  """ Check if the sesssion state has a solution plan """
  return "solution_plan" in st.session_state \
    and st.session_state.solution_plan is not None

def main():
    # Initialize the streamlit app
    render_app()


if __name__ == "__main__":
    main()
