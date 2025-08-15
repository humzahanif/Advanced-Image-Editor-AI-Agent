import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain.schema import BaseMessage
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import cv2 
import numpy as np
import io
import base64
from typing import Optional, List, Dict, Any
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import pandas as pd

# Configure page
st.set_page_config(
    page_title="🎨 Advanced Image Editor AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .feature-box {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .result-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f1f8e9;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class ImageAnalysisTool(BaseTool):
    name: str = "image_analyzer"
    description: str = "Analyzes images using Gemini vision model for detailed descriptions, object detection, and insights"
    
    def _run(self, image_data: str, prompt: str = "Analyze this image in detail") -> str:
        try:
            genai.configure(api_key=st.session_state.api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(io.BytesIO(image_bytes))
            response = model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            return f"Error analyzing image: {str(e)}"


class ImageEnhancementTool(BaseTool):
    name: str = "image_enhancer"
    description: str = "Enhances images with various filters and adjustments"
    
    def _run(self, image_data: str, enhancement_type: str = "auto") -> Dict[str, Any]:
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(io.BytesIO(image_bytes))
            enhanced_images = {}
            if enhancement_type == "auto" or enhancement_type == "all":
                enhancer = ImageEnhance.Brightness(image)
                enhanced_images['brightness'] = enhancer.enhance(1.2)
                enhancer = ImageEnhance.Contrast(image)
                enhanced_images['contrast'] = enhancer.enhance(1.3)
                enhancer = ImageEnhance.Sharpness(image)
                enhanced_images['sharpness'] = enhancer.enhance(1.5)
                enhanced_images['blur'] = image.filter(ImageFilter.GaussianBlur(1))
                enhanced_images['sharpen'] = image.filter(ImageFilter.SHARPEN)
                enhanced_images['edge_enhance'] = image.filter(ImageFilter.EDGE_ENHANCE)
            return enhanced_images
        except Exception as e:
            return {"error": f"Error enhancing image: {str(e)}"}


class ColorAnalysisTool(BaseTool):
    name: str = "color_analyzer"
    description: str = "Analyzes color composition and palette of images"
    
    def _run(self, image_data: str, n_colors: int = 5) -> Dict[str, Any]:
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_rgb = image.convert('RGB')
            data = np.array(image_rgb).reshape((-1, 3))
            kmeans = KMeans(n_clusters=n_colors, random_state=42)
            kmeans.fit(data)
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            color_counts = np.bincount(labels)
            color_percentages = color_counts / len(labels) * 100
            palette_info = []
            for color, percentage in zip(colors, color_percentages):
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                palette_info.append({
                    'color': color.tolist(),
                    'hex': hex_color,
                    'percentage': round(percentage, 2)
                })
            return {
                'palette': palette_info,
                'dominant_color': colors[np.argmax(color_percentages)].tolist(),
                'total_colors': len(np.unique(data.view(np.dtype((np.void, data.dtype.itemsize*data.shape[1])))))
            }
        except Exception as e:
            return {"error": f"Error analyzing colors: {str(e)}"}


class ImageMetadataTool(BaseTool):
    name: str = "metadata_extractor"
    description: str = "Extracts metadata and technical information from images"
    
    def _run(self, image_data: str) -> Dict[str, Any]:
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(io.BytesIO(image_bytes))
            metadata = {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'aspect_ratio': round(image.width / image.height, 2),
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                metadata['exif'] = {k: str(v) for k, v in exif.items() if k < 50000}
            metadata['file_size_bytes'] = len(image_bytes)
            metadata['file_size_kb'] = round(len(image_bytes) / 1024, 2)
            return metadata
        except Exception as e:
            return {"error": f"Error extracting metadata: {str(e)}"}

class ImageGenerationTool(BaseTool):
    name: str = "image_generator"
    description: str = "Generates images from text prompts using Gemini's image generation model"

    def _run(self, prompt: str, size: str = "1024x1024") -> str:
        try:
            genai.configure(api_key=st.session_state.api_key)
            model = genai.GenerativeModel('imagen-3.0')  # Replace with actual image model name
            result = model.generate_image(prompt=prompt, size=size)
            image_data = result.images[0]  # assuming first image
            img_b64 = base64.b64encode(image_data).decode()
            return f"data:image/png;base64,{img_b64}"
        except Exception as e:
            return f"Error generating image: {str(e)}"

def setup_agent():
    """Setup the LangChain agent with image processing tools"""
    if not st.session_state.api_key:
        st.error("Please enter your Google AI API key in the sidebar.")
        return None
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=st.session_state.api_key,
        temperature=0.7
    )
    
    # Initialize tools
    tools = [
        ImageAnalysisTool(),
        ImageEnhancementTool(),
        ColorAnalysisTool(),
        ImageMetadataTool()
    ]
    
    # Create agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def create_color_palette_viz(color_data):
    """Create a color palette visualization"""
    if 'error' in color_data:
        return None
    
    colors = [item['hex'] for item in color_data['palette']]
    percentages = [item['percentage'] for item in color_data['palette']]
    
    fig = go.Figure(data=[go.Bar(
        x=colors,
        y=percentages,
        marker_color=colors,
        text=[f"{p}%" for p in percentages],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Dominant Colors",
        xaxis_title="Colors",
        yaxis_title="Percentage",
        showlegend=False
    )
    
    return fig

def apply_opencv_effects(image, effect_type):
    """Apply OpenCV effects to image"""
    # Convert PIL to OpenCV
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    if effect_type == "edge_detection":
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(edges, 'L')
    elif effect_type == "cartoon":
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(cv_image, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
    elif effect_type == "vintage":
        # Apply vintage effect
        vintage = cv2.applyColorMap(cv_image, cv2.COLORMAP_AUTUMN)
        return Image.fromarray(cv2.cvtColor(vintage, cv2.COLOR_BGR2RGB))
    
    return image

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Advanced Image AI Agent</h1>
        <p>Powered by LangChain + Gemini 2.5 Flash + Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔑 Configuration")
        api_key = st.text_input("Google AI API Key", type="password", value=st.session_state.api_key)
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        st.header("📋 Features")
        st.markdown("""
        - **AI Image Analysis** - Detailed descriptions
        - **Object Detection** - Identify objects and scenes
        - **Color Analysis** - Extract color palettes
        - **Image Enhancement** - Auto-enhance images
        - **Style Transfer** - Apply artistic effects
        - **Metadata Extraction** - Technical information
        - **Batch Processing** - Process multiple images
        - **Export Results** - Download processed images
        """)
        
        if st.button("🗑️ Clear History"):
            st.session_state.processed_images = []
            st.session_state.chat_history = []
            st.rerun()
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Upload", "🎨 Processing", "📊 Analytics", "💬 AI Chat"])
    
    with tab1:
        st.header("Upload Images")
        
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            cols = st.columns(min(3, len(uploaded_files)))
            for idx, uploaded_file in enumerate(uploaded_files):
                with cols[idx % 3]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                    
                    if st.button(f"Process {uploaded_file.name}", key=f"process_{idx}"):
                        st.session_state.processed_images.append({
                            'name': uploaded_file.name,
                            'image': image,
                            'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.success(f"Added {uploaded_file.name} to processing queue!")
    
    with tab2:
        st.header("Image Processing")
        
        if not st.session_state.processed_images:
            st.warning("Please upload and add images to the processing queue first.")
            return
        
        # Select image to process
        selected_img_idx = st.selectbox(
            "Select image to process:",
            range(len(st.session_state.processed_images)),
            format_func=lambda x: st.session_state.processed_images[x]['name']
        )
        
        if selected_img_idx is not None:
            selected_image = st.session_state.processed_images[selected_img_idx]['image']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(selected_image, use_container_width=True)
                
                # Processing options
                st.subheader("Enhancement Options")
                
                brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
                contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
                
                filter_option = st.selectbox(
                    "Apply Filter:",
                    ["None", "Blur", "Sharpen", "Edge Enhance", "Smooth", "Detail"]
                )
                
                effect_option = st.selectbox(
                    "Apply Effect:",
                    ["None", "Edge Detection", "Cartoon", "Vintage", "Grayscale", "Sepia"]
                )
            
            with col2:
                st.subheader("Processed Image")
                
                # Apply enhancements
                processed_image = selected_image.copy()
                
                # Brightness
                enhancer = ImageEnhance.Brightness(processed_image)
                processed_image = enhancer.enhance(brightness)
                
                # Contrast
                enhancer = ImageEnhance.Contrast(processed_image)
                processed_image = enhancer.enhance(contrast)
                
                # Saturation
                enhancer = ImageEnhance.Color(processed_image)
                processed_image = enhancer.enhance(saturation)
                
                # Apply filter
                if filter_option != "None":
                    filter_map = {
                        "Blur": ImageFilter.BLUR,
                        "Sharpen": ImageFilter.SHARPEN,
                        "Edge Enhance": ImageFilter.EDGE_ENHANCE,
                        "Smooth": ImageFilter.SMOOTH,
                        "Detail": ImageFilter.DETAIL
                    }
                    processed_image = processed_image.filter(filter_map[filter_option])
                
                # Apply effect
                if effect_option != "None":
                    if effect_option == "Grayscale":
                        processed_image = ImageOps.grayscale(processed_image)
                        processed_image = processed_image.convert('RGB')
                    elif effect_option == "Sepia":
                        processed_image = ImageOps.colorize(
                            ImageOps.grayscale(processed_image), 
                            '#704214', '#C0A882'
                        )
                    else:
                        effect_map = {
                            "Edge Detection": "edge_detection",
                            "Cartoon": "cartoon",
                            "Vintage": "vintage"
                        }
                        processed_image = apply_opencv_effects(
                            processed_image, 
                            effect_map[effect_option]
                        )
                
                st.image(processed_image, use_container_width=True)
                
                # Download processed image
                if st.button("💾 Download Processed Image"):
                    buffer = io.BytesIO()
                    processed_image.save(buffer, format='PNG')
                    st.download_button(
                        label="Click to Download",
                        data=buffer.getvalue(),
                        file_name=f"processed_{selected_img_idx}.png",
                        mime="image/png"
                    )
    
    with tab3:
        st.header("Image Analytics")
        
        if not st.session_state.processed_images:
            st.warning("Please upload images first.")
            return
        
        # Select image for analysis
        selected_img_idx = st.selectbox(
            "Select image for analysis:",
            range(len(st.session_state.processed_images)),
            format_func=lambda x: st.session_state.processed_images[x]['name'],
            key="analytics_select"
        )
        
        if selected_img_idx is not None:
            image = st.session_state.processed_images[selected_img_idx]['image']
            image_b64 = image_to_base64(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Image Preview")
                st.image(image, use_container_width=True)
                
                # Metadata analysis
                if st.button("📊 Analyze Metadata"):
                    metadata_tool = ImageMetadataTool()
                    metadata = metadata_tool._run(image_b64)
                    
                    with st.expander("📋 Technical Information"):
                        for key, value in metadata.items():
                            if key != 'exif':
                                st.write(f"**{key.replace('_', ' ').title()}**: {value}")
            
            with col2:
                # Color analysis
                if st.button("🎨 Analyze Colors"):
                    color_tool = ColorAnalysisTool()
                    color_data = color_tool._run(image_b64, n_colors=6)
                    
                    if 'error' not in color_data:
                        st.subheader("Color Palette")
                        
                        # Create color swatches
                        palette_cols = st.columns(len(color_data['palette']))
                        for i, color_info in enumerate(color_data['palette']):
                            with palette_cols[i]:
                                st.color_picker(
                                    f"{color_info['percentage']:.1f}%",
                                    color_info['hex'],
                                    disabled=True,
                                    key=f"color_{i}"
                                )
                        
                        # Color chart
                        fig = create_color_palette_viz(color_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.write(f"**Dominant Color**: {color_data['dominant_color']}")
                        st.write(f"**Total Unique Colors**: {color_data['total_colors']:,}")
    
    with tab4:
        st.header("AI Image Chat")
        
        if not st.session_state.api_key:
            st.error("Please enter your Google AI API key to use the chat feature.")
            return
        
        # Chat interface
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "image" in message:
                    st.image(message["image"], width=200)
        
        # Chat input
        user_input = st.chat_input("Ask me about your images...")
        
        if user_input and st.session_state.processed_images:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get latest image
            latest_image = st.session_state.processed_images[-1]['image']
            image_b64 = image_to_base64(latest_image)
            
            # Generate AI response
            try:
                analysis_tool = ImageAnalysisTool()
                ai_response = analysis_tool._run(image_b64, user_input)
                
                # Add AI response
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": ai_response,
                    "image": latest_image
                })
                
                with st.chat_message("assistant"):
                    st.write(ai_response)
                    st.image(latest_image, width=200)
                    
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()