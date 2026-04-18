import streamlit as st
import tempfile
from attentionx_backend import process_video_full_pipeline

st.set_page_config(page_title="AttentionX AI", layout="wide")

st.title("🎬 AttentionX AI")
st.subheader("AI-powered video to short clips")

st.divider()

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

video_path = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(uploaded_file)
    st.success("Video ready")

if video_path:
    if st.button("🚀 Generate Clips"):

        with st.spinner("Processing video (1-3 min)..."):
            results = process_video_full_pipeline(video_path)

        st.success("Done!")

        col1, col2 = st.columns(2)
        col1.metric("Peaks", len(results["peaks"]))
        col2.metric("Clips", len(results["clips"]))

        for clip in results["clips"]:
            st.video(clip["path"])
            st.caption(f"Clip {clip['clip_number']}")