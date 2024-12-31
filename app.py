import streamlit as st
import PyPDF2
import google.generativeai as genai
from pathlib import Path
import os
from config import GOOGLE_API_KEY
import fitz  # PyMuPDF for better PDF parsing
import tempfile
from typing import List, Dict
import json



def configure_gemini():
    """Configure the Gemini API with your credentials"""
    if not GOOGLE_API_KEY:
        print("Please set your GOOGLE_API_KEY")
        exit()  # Stop execution if API key is not found

    # Configure Gemini API with the provided key
    genai.configure(api_key=GOOGLE_API_KEY)

    # Return the configured Gemini model
    return genai.GenerativeModel('gemini-pro')


def extract_text_from_pdf(pdf_file, parser_choice="PyMuPDF"):
    """Extract text content from uploaded PDF file using selected parser"""
    try:
        if parser_choice == "PyMuPDF":
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name

            # Extract text using PyMuPDF
            doc = fitz.open(tmp_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            os.unlink(tmp_path)
            return text
        else:
            # Fallback to PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None


def generate_summary(model, text):
    """Generate a summary of the text using Gemini API"""
    prompt = f"""
    Please provide a comprehensive summary of the following text, including:
    1. Main topics and key points
    2. Important findings or conclusions
    3. Notable details or examples

    Text: {text[:5000]}  # Limiting text length to avoid token limits
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None


def generate_custom_qa(model, text, question_type, num_questions=5):
    """Generate custom questions and answers based on selected type"""
    prompt_templates = {
        "factual": "Generate {n} factual questions that test knowledge of specific information from the text.",
        "conceptual": "Generate {n} conceptual questions that require understanding of main ideas and themes.",
        "analytical": "Generate {n} analytical questions that require critical thinking and analysis of the content.",
        "application": "Generate {n} questions that ask how the concepts from the text can be applied to real-world situations."
    }

    prompt = f"""
    {prompt_templates[question_type].format(n=num_questions)}
    Format the response as a list of dictionaries with 'question' and 'answer' keys.
    Make answers detailed and comprehensive.

    Text: {text[:5000]}
    """

    try:
        response = model.generate_content(prompt)
        return eval(response.text)  # Note: In production, use safer parsing
    except Exception as e:
        st.error(f"Error generating Q&A: {str(e)}")
        return None


def process_multiple_pdfs(files, model):
    """Process multiple PDF files and generate combined analysis"""
    results = []

    for file in files:
        text = extract_text_from_pdf(file)
        if text:
            summary = generate_summary(model, text)
            qa_pairs = generate_custom_qa(model, text, "factual", 3)  # Generate 3 questions per file

            results.append({
                "filename": file.name,
                "summary": summary,
                "qa_pairs": qa_pairs
            })

    return results


def save_analysis_results(results):
    """Save analysis results as JSON"""
    return json.dumps(results, indent=2)


def main():
    st.title("Enhanced PDF Analysis with Gemini AI")
    st.write("Upload PDF files for comprehensive analysis and Q&A generation")

    # Initialize Gemini model
    model = configure_gemini()

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    parser_choice = st.sidebar.selectbox(
        "PDF Parser",
        ["PyMuPDF", "PyPDF2"],
        help="PyMuPDF generally provides better results but may be slower"
    )

    # Multiple file upload
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Analysis options
        analysis_type = st.radio(
            "Choose analysis type",
            ["Single File Analysis", "Batch Analysis"]
        )

        if analysis_type == "Single File Analysis":
            # Process single file with more options
            selected_file = st.selectbox("Select file to analyze", uploaded_files)

            with st.spinner("Processing PDF..."):
                text = extract_text_from_pdf(selected_file, parser_choice)

                if text:
                    st.success("PDF processed successfully!")

                    # Text preview
                    with st.expander("View extracted text"):
                        st.text(text[:1000] + "..." if len(text) > 1000 else text)

                    # Generate summary
                    if st.button("Generate Summary"):
                        with st.spinner("Generating summary..."):
                            summary = generate_summary(model, text)
                            if summary:
                                st.subheader("Document Summary")
                                st.write(summary)

                    # Q&A Generation
                    st.subheader("Generate Questions & Answers")
                    question_type = st.selectbox(
                        "Question Type",
                        ["factual", "conceptual", "analytical", "application"]
                    )
                    num_questions = st.slider("Number of questions", 1, 10, 5)

                    if st.button("Generate Q&A"):
                        with st.spinner("Generating questions and answers..."):
                            qa_pairs = generate_custom_qa(model, text, question_type, num_questions)

                            if qa_pairs:
                                for i, qa in enumerate(qa_pairs, 1):
                                    with st.expander(f"Q{i}: {qa['question']}", expanded=True):
                                        st.write(qa['answer'])

                                # Download Q&A
                                qa_text = "\n\n".join([f"Q{i}: {qa['question']}\nA: {qa['answer']}"
                                                       for i, qa in enumerate(qa_pairs, 1)])
                                st.download_button(
                                    label="Download Q&A",
                                    data=qa_text,
                                    file_name=f"qa_results_{selected_file.name}.txt",
                                    mime="text/plain"
                                )

        else:
            # Batch analysis
            st.write(f"Processing {len(uploaded_files)} files...")

            if st.button("Start Batch Analysis"):
                with st.spinner("Processing files..."):
                    results = process_multiple_pdfs(uploaded_files, model)

                    if results:
                        st.success("Batch analysis complete!")

                        # Display results
                        for result in results:
                            with st.expander(f"Results for {result['filename']}", expanded=True):
                                st.subheader("Summary")
                                st.write(result['summary'])

                                st.subheader("Key Questions & Answers")
                                for i, qa in enumerate(result['qa_pairs'], 1):
                                    st.write(f"Q{i}: {qa['question']}")
                                    st.write(f"A: {qa['answer']}")
                                    st.write("---")

                        # Download all results
                        json_results = save_analysis_results(results)
                        st.download_button(
                            label="Download All Results",
                            data=json_results,
                            file_name="batch_analysis_results.json",
                            mime="application/json"
                        )


if __name__ == "__main__":
    main()