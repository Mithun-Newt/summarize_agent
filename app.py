import os
import validators
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)
###code
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader
)

load_dotenv()

# Streamlit App
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL Content")

# Sidebar
with st.sidebar:
    hf_api_key = st.text_input(
        "Hugging Face API Token",
        value="",
        type="password"
    )

# URL Input
generic_url = st.text_input(
    "URL",
    label_visibility="collapsed"
)

# Prompt
prompt_template = """
Provide a summary of the following content in approximately 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

# Button
if st.button("Summarize the Content"):

    if not hf_api_key.strip():
        st.error("Please enter your Hugging Face API Token")

    elif not generic_url.strip():
        st.error("Please enter a URL")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL")

    else:

        try:

            with st.spinner("Loading content and generating summary..."):

                # Hugging Face Endpoint
                llm_endpoint = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.1-8B-Instruct",
                    huggingfacehub_api_token=hf_api_key,
                    max_new_tokens=256,
                    temperature=0.7
                )

                # Chat Wrapper
                llm = ChatHuggingFace(
                    llm=llm_endpoint
                )

                # Load Content
                if (
                    "youtube.com" in generic_url
                    or
                    "youtu.be" in generic_url
                ):

                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False
                    )

                else:

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent":
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                        }
                    )

                docs = loader.load()

                if len(docs) == 0:
                    st.error(
                        "No content could be extracted from the URL."
                    )
                    st.stop()

                # Summarization Chain
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                summary = chain.run(
                    docs
                )

                st.success(
                    "Summary Generated Successfully"
                )

                st.write(summary)

        except Exception as e:

            st.exception(e)
