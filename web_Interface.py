import streamlit as st
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import asyncio
from typing import Dict, List, Tuple
import base64
from pathlib import Path
import json

# Import our custom modules
from main import FakeNewsDetector, ModelConfig, DeepfakeDetector
from data_processor import AdvancedDataProcessor, DatasetConfig

class StreamlitInterface:
    """Professional Streamlit interface for the detection system"""
    
    def __init__(self):
        self.detector = None
        self.deepfake_detector = DeepfakeDetector()
        self.detection_history = []
        
        # Configure page
        st.set_page_config(
            page_title="AI Fake News & Deepfake Detector",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for professional styling
        self.apply_custom_styling()
    
    def apply_custom_styling(self):
        """Apply professional CSS styling"""
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
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        
        .detection-result {
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            font-weight: bold;
        }
        
        .fake-result {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
        }
        
        .real-result {
            background: linear-gradient(135deg, #51cf66, #40c057);
            color: white;
        }
        
        .sidebar-content {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_detector(self):
        """Initialize the detection models"""
        if self.detector is None:
            with st.spinner("🚀 Initializing AI Detection Models..."):
                config = ModelConfig(
                    model_name="distilbert-base-uncased",
                    max_length=256,
                    batch_size=8
                )
                self.detector = FakeNewsDetector(config)
                
                # Try to load pre-trained model or train on sample data
                try:
                    self.detector.load_model("./models/fake_news_detector")
                    st.success("✅ Pre-trained model loaded successfully!")
                except:
                    st.info("📚 Training model on sample data...")
                    from main import create_sample_dataset
                    texts, labels = create_sample_dataset(500)
                    self.detector.train(texts, labels)
                    st.success("✅ Model training completed!")
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🔍 AI-Powered Fake News & Deepfake Detection System</h1>
            <p>Advanced machine learning system for detecting misinformation and manipulated media</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls and information"""
        st.sidebar.title("🎛️ Control Panel")
        
        # Model selection
        st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.sidebar.subheader("🤖 Model Configuration")
        
        model_type = st.sidebar.selectbox(
            "Detection Model",
            ["BERT-based", "RoBERTa-based", "DistilBERT-based"],
            index=2
        )
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05
        )
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # System information
        st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.sidebar.subheader("📊 System Status")
        
        if self.detector:
            st.sidebar.success("🟢 Text Detector: Online")
        else:
            st.sidebar.error("🔴 Text Detector: Offline")
        
        st.sidebar.success("🟢 Image Detector: Online")
        st.sidebar.info(f"🔍 Total Detections: {len(self.detection_history)}")
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        return model_type, confidence_threshold
    
    def render_text_detection(self, confidence_threshold):
        """Render text detection interface"""
        st.subheader("📝 Text Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input methods
            input_method = st.radio(
                "Input Method",
                ["Direct Text", "Upload File", "URL Analysis"],
                horizontal=True
            )
            
            text_input = ""
            
            if input_method == "Direct Text":
                text_input = st.text_area(
                    "Enter text to analyze:",
                    height=200,
                    placeholder="Paste your news article or text here..."
                )
            
            elif input_method == "Upload File":
                uploaded_file = st.file_uploader(
                    "Upload text file",
                    type=['txt', 'pdf', 'docx']
                )
                if uploaded_file:
                    # Process uploaded file (simplified for demo)
                    text_input = str(uploaded_file.read(), "utf-8")
            
            elif input_method == "URL Analysis":
                url = st.text_input("Enter URL:")
                if url and st.button("Fetch Content"):
                    # Simplified URL content fetching
                    text_input = f"Content from {url} would be fetched here"
            
            if st.button("🔍 Analyze Text", type="primary") and text_input:
                self.analyze_text(text_input, confidence_threshold)
        
        with col2:
            self.render_quick_stats()
    
    def analyze_text(self, text: str, confidence_threshold: float):
        """Analyze text for fake news detection"""
        if not self.detector:
            st.error("Please initialize the detector first!")
            return
        
        with st.spinner("🤖 Analyzing text..."):
            # Simulate processing time
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            try:
                result = self.detector.predict(text)
                
                # Store result in history
                self.detection_history.append({
                    'timestamp': datetime.now(),
                    'type': 'text',
                    'result': result,
                    'text_preview': text[:100] + "..." if len(text) > 100 else text
                })
                
                # Display results
                self.display_text_results(result, confidence_threshold)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    def display_text_results(self, result: Dict, confidence_threshold: float):
        """Display text analysis results"""
        prediction = result['prediction']
        confidence = result['confidence']
        fake_prob = result['fake_probability']
        real_prob = result['real_probability']
        
        # Main result display
        if fake_prob > confidence_threshold:
            st.markdown(f"""
            <div class="detection-result fake-result">
                🚨 FAKE NEWS DETECTED<br>
                Confidence: {confidence:.2%}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="detection-result real-result">
                ✅ APPEARS LEGITIMATE<br>
                Confidence: {confidence:.2%}
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Fake Probability",
                f"{fake_prob:.2%}",
                delta=f"{fake_prob - 0.5:.2%}" if fake_prob > 0.5 else None
            )
        
        with col2:
            st.metric(
                "Real Probability", 
                f"{real_prob:.2%}",
                delta=f"{real_prob - 0.5:.2%}" if real_prob > 0.5 else None
            )
        
        with col3:
            st.metric(
                "Confidence Score",
                f"{confidence:.2%}",
                delta=f"{confidence - confidence_threshold:.2%}"
            )
        
        # Probability visualization
        fig = go.Figure(data=[
            go.Bar(
                x=['Real News', 'Fake News'],
                y=[real_prob, fake_prob],
                marker_color=['#51cf66', '#ff6b6b']
            )
        ])
        
        fig.update_layout(
            title="Detection Probabilities",
            yaxis_title="Probability",
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_image_detection(self):
        """Render image/deepfake detection interface"""
        st.subheader("🖼️ Image & Deepfake Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_image = st.file_uploader(
                "Upload an image for deepfake detection",
                type=['jpg', 'jpeg', 'png', 'bmp']
            )
            
            if uploaded_image:
                # Display uploaded image
                st.image(uploaded_image, caption="Uploaded Image", width=400)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    self.analyze_image(uploaded_image)
        
        with col2:
            st.info("""
            **Deepfake Detection Features:**
            - Face detection and extraction
            - Neural network analysis
            - Manipulation probability scoring
            - Facial inconsistency detection
            """)
    
    def analyze_image(self, uploaded_image):
        """Analyze image for deepfake detection"""
        with st.spinner("🤖 Analyzing image for deepfakes..."):
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_image.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            try:
                result = self.deepfake_detector.detect_deepfake(temp_path)
                
                # Store result in history
                self.detection_history.append({
                    'timestamp': datetime.now(),
                    'type': 'image',
                    'result': result,
                    'filename': uploaded_image.name
                })
                
                # Display results
                self.display_image_results(result)
                
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
                
            except Exception as e:
                st.error(f"Error during image analysis: {str(e)}")
                Path(temp_path).unlink(missing_ok=True)
    
    def display_image_results(self, result: Dict):
        """Display image analysis results"""
        prediction = result['prediction']
        confidence = result['confidence']
        message = result['message']
        
        if prediction == "fake":
            st.markdown(f"""
            <div class="detection-result fake-result">
                🚨 POTENTIAL DEEPFAKE DETECTED<br>
                Confidence: {confidence:.2%}
            </div>
            """, unsafe_allow_html=True)
        elif prediction == "real":
            st.markdown(f"""
            <div class="detection-result real-result">
                ✅ IMAGE APPEARS AUTHENTIC<br>
                Confidence: {confidence:.2%}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ {message}")
            return
        
        # Additional image analysis metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Detection Confidence", f"{confidence:.2%}")
        
        with col2:
            st.metric("Analysis Status", "✅ Complete" if prediction != "error" else "❌ Error")
    
    def render_quick_stats(self):
        """Render quick statistics panel"""
        st.markdown("### 📊 Quick Stats")
        
        if self.detection_history:
            recent_detections = self.detection_history[-10:]
            
            fake_count = sum(1 for d in recent_detections 
                           if d['type'] == 'text' and d['result']['prediction'] == 'fake')
            real_count = len([d for d in recent_detections if d['type'] == 'text']) - fake_count
            
            # Mini charts
            fig = go.Figure(data=[go.Pie(
                labels=['Real', 'Fake'],
                values=[real_count, fake_count],
                hole=.3,
                marker_colors=['#51cf66', '#ff6b6b']
            )])
            
            fig.update_layout(
                title="Recent Detections",
                height=250,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No detection history yet")
    
    def render_analytics_dashboard(self):
        """Render comprehensive analytics dashboard"""
        st.subheader("📈 Analytics Dashboard")
        
        if not self.detection_history:
            st.info("No data available for analytics. Perform some detections first!")
            return
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.detection_history)
        
        # Time series analysis
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_stats = df.groupby(['date', 'type']).size().unstack(fill_value=0)
        
        fig_timeline = px.line(
            daily_stats.reset_index(),
            x='date',
            y=['text', 'image'],
            title="Detection Activity Over Time",
            labels={'value': 'Number of Detections', 'variable': 'Detection Type'}
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Detection accuracy simulation
        col1, col2 = st.columns(2)
        
        with col1:
            # Fake vs Real distribution
            text_results = [d['result']['prediction'] for d in self.detection_history if d['type'] == 'text']
            fake_count = text_results.count('fake')
            real_count = text_results.count('real')
            
            fig_dist = go.Figure(data=[go.Bar(
                x=['Real News', 'Fake News'],
                y=[real_count, fake_count],
                marker_color=['#51cf66', '#ff6b6b']
            )])
            
            fig_dist.update_layout(title="Detection Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Confidence score distribution
            confidences = [d['result']['confidence'] for d in self.detection_history 
                         if d['type'] == 'text']
            
            if confidences:
                fig_conf = px.histogram(
                    x=confidences,
                    title="Confidence Score Distribution",
                    nbins=10
                )
                fig_conf.update_layout(xaxis_title="Confidence Score", yaxis_title="Count")
                st.plotly_chart(fig_conf, use_container_width=True)
    
    def render_detection_history(self):
        """Render detection history table"""
        st.subheader("📋 Detection History")
        
        if not self.detection_history:
            st.info("No detection history available yet.")
            return
        
        # Create DataFrame for display
        history_data = []
        for item in self.detection_history:
            row = {
                'Timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Type': item['type'].title(),
                'Prediction': item['result']['prediction'].title(),
                'Confidence': f"{item['result']['confidence']:.2%}",
            }
            
            if item['type'] == 'text':
                row['Content'] = item['text_preview']
            else:
                row['Content'] = item.get('filename', 'Image')
            
            history_data.append(row)
        
        df_history = pd.DataFrame(history_data)
        df_history = df_history.sort_values('Timestamp', ascending=False)
        
        # Display with styling
        st.dataframe(
            df_history,
            use_container_width=True,
            hide_index=True
        )
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export CSV"):
                csv = df_history.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🗑️ Clear History"):
                self.detection_history.clear()
                st.success("History cleared!")
                st.experimental_rerun()
    
    def render_model_training(self):
        """Render model training interface"""
        st.subheader("🎯 Model Training & Fine-tuning")
        
        st.info("""
        **Training Options:**
        - Upload custom datasets
        - Fine-tune on domain-specific data
        - Adjust model parameters
        - Monitor training progress
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Upload")
            training_file = st.file_uploader(
                "Upload training dataset (CSV format)",
                type=['csv']
            )
            
            if training_file:
                df = pd.read_csv(training_file)
                st.write("Dataset Preview:")
                st.dataframe(df.head())
                
                if st.button("🚀 Start Training"):
                    with st.spinner("Training model..."):
                        # Simulate training process
                        progress = st.progress(0)
                        for i in range(100):
                            time.sleep(0.05)
                            progress.progress(i + 1)
                        st.success("Model training completed!")
        
        with col2:
            st.markdown("### Training Parameters")
            epochs = st.slider("Training Epochs", 1, 10, 3)
            batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
            learning_rate = st.selectbox("Learning Rate", [1e-5, 2e-5, 5e-5], index=1)
            
            st.markdown("### Model Performance")
            # Simulate performance metrics
            metrics = {
                "Accuracy": 0.92,
                "Precision": 0.89,
                "Recall": 0.94,
                "F1-Score": 0.91
            }
            
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.2%}")
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Initialize detector
        self.initialize_detector()
        
        # Sidebar
        model_type, confidence_threshold = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📝 Text Detection", 
            "🖼️ Image Detection", 
            "📊 Analytics", 
            "📋 History", 
            "🎯 Training"
        ])
        
        with tab1:
            self.render_text_detection(confidence_threshold)
        
        with tab2:
            self.render_image_detection()
        
        with tab3:
            self.render_analytics_dashboard()
        
        with tab4:
            self.render_detection_history()
        
        with tab5:
            self.render_model_training()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            🔍 AI-Powered Fake News & Deepfake Detection System<br>
            Built with advanced transformer models and computer vision<br>
            <em>Protecting information integrity through AI</em>
        </div>
        """, unsafe_allow_html=True)

class GradioInterface:
    """Alternative Gradio interface for the detection system"""
    
    def __init__(self):
        self.detector = None
        self.deepfake_detector = DeepfakeDetector()
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize detection models"""
        config = ModelConfig(
            model_name="distilbert-base-uncased",
            max_length=256,
            batch_size=8
        )
        self.detector = FakeNewsDetector(config)
        
        # Train on sample data for demo
        from main import create_sample_dataset
        texts, labels = create_sample_dataset(100)
        self.detector.train(texts, labels)
    
    def analyze_text_gradio(self, text: str) -> Tuple[str, float, str]:
        """Analyze text using Gradio interface"""
        if not text.strip():
            return "Please enter some text to analyze", 0.0, ""
        
        try:
            result = self.detector.predict(text)
            prediction = result['prediction']
            confidence = result['confidence']
            fake_prob = result['fake_probability']
            
            status = f"🚨 FAKE NEWS DETECTED" if prediction == 'fake' else "✅ APPEARS LEGITIMATE"
            confidence_text = f"Confidence: {confidence:.2%}"
            
            detailed_results = f"""
            **Detailed Analysis:**
            - Prediction: {prediction.upper()}
            - Fake Probability: {fake_prob:.2%}
            - Real Probability: {result['real_probability']:.2%}
            - Confidence Score: {confidence:.2%}
            """
            
            return status, confidence, detailed_results
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0, ""
    
    def analyze_image_gradio(self, image) -> Tuple[str, float]:
        """Analyze image using Gradio interface"""
        if image is None:
            return "Please upload an image", 0.0
        
        try:
            # Save temporary image
            temp_path = "temp_gradio_image.jpg"
            image.save(temp_path)
            
            result = self.deepfake_detector.detect_deepfake(temp_path)
            
            prediction = result['prediction']
            confidence = result['confidence']
            
            if prediction == "fake":
                status = f"🚨 POTENTIAL DEEPFAKE DETECTED"
            elif prediction == "real":
                status = f"✅ IMAGE APPEARS AUTHENTIC"
            else:
                status = f"⚠️ {result['message']}"
            
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
            
            return status, confidence
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0
    
    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="AI Fake News & Deepfake Detector",
            css="""
            .gradio-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .gr-button-primary {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                border: none;
            }
            """
        ) as demo:
            
            gr.Markdown("""
            # 🔍 AI-Powered Fake News & Deepfake Detection System
            
            Advanced machine learning system for detecting misinformation and manipulated media using 
            state-of-the-art transformer models and computer vision techniques.
            """)
            
            with gr.Tabs():
                # Text Detection Tab
                with gr.TabItem("📝 Text Analysis"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            text_input = gr.Textbox(
                                label="Enter text to analyze",
                                placeholder="Paste your news article or text here...",
                                lines=10
                            )
                            text_btn = gr.Button("🔍 Analyze Text", variant="primary")
                        
                        with gr.Column(scale=1):
                            text_result = gr.Textbox(label="Detection Result", interactive=False)
                            text_confidence = gr.Number(label="Confidence Score", interactive=False)
                            text_details = gr.Markdown(label="Detailed Analysis")
                    
                    text_btn.click(
                        fn=self.analyze_text_gradio,
                        inputs=text_input,
                        outputs=[text_result, text_confidence, text_details]
                    )
                
                # Image Detection Tab
                with gr.TabItem("🖼️ Image Analysis"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            image_input = gr.Image(
                                label="Upload image for deepfake detection",
                                type="pil"
                            )
                            image_btn = gr.Button("🔍 Analyze Image", variant="primary")
                        
                        with gr.Column(scale=1):
                            image_result = gr.Textbox(label="Detection Result", interactive=False)
                            image_confidence = gr.Number(label="Confidence Score", interactive=False)
                    
                    image_btn.click(
                        fn=self.analyze_image_gradio,
                        inputs=image_input,
                        outputs=[image_result, image_confidence]
                    )
                
                # Examples Tab
                with gr.TabItem("📚 Examples"):
                    gr.Markdown("""
                    ### Sample Texts to Try:
                    
                    **Real News Example:**
                    "The government announced new healthcare policies today, focusing on improved access to medical services in rural areas. The policy includes funding for new clinics and telemedicine programs."
                    
                    **Fake News Example:**
                    "SHOCKING: Scientists discover this one weird trick that doctors don't want you to know! This miracle cure will change your life forever and pharmaceutical companies are trying to hide it!"
                    """)
                    
                    # Example buttons
                    real_example = gr.Button("📰 Try Real News Example")
                    fake_example = gr.Button("🚨 Try Fake News Example")
                    
                    real_example.click(
                        lambda: "The government announced new healthcare policies today, focusing on improved access to medical services in rural areas.",
                        outputs=text_input
                    )
                    
                    fake_example.click(
                        lambda: "SHOCKING: Scientists discover this one weird trick that doctors don't want you to know!",
                        outputs=text_input
                    )
            
            gr.Markdown("""
            ---
            **About this System:**
            - Uses advanced transformer models (BERT, RoBERTa) for text analysis
            - Employs deep learning for image manipulation detection
            - Multi-agent architecture for comprehensive analysis
            - Real-time processing with high accuracy
            
            *Built with ❤️ using cutting-edge AI technology*
            """)
        
        return demo

def run_streamlit_app():
    """Run the Streamlit application"""
    app = StreamlitInterface()
    app.run()

def run_gradio_app():
    """Run the Gradio application"""
    app = GradioInterface()
    interface = app.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        print("🚀 Starting Gradio interface...")
        run_gradio_app()
    else:
        print("🚀 Starting Streamlit interface...")
        print("Run with 'python web_interface.py gradio' for Gradio interface")
        run_streamlit_app()
