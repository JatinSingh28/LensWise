import streamlit as st
from lensWise import LensWise


def main():
    st.title('Lens Wise')
    
    if "started" not in st.session_state:
        st.session_state.started = False
        
    if not st.session_state.started:
        st.session_state.lenswise = LensWise()
        st.session_state.started = True
        
    if 'capturing' not in st.session_state:
        st.session_state.capturing = False
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = None
        
    st.title("Live Video Feed with Q&A")

    frame_display = st.empty()
    
    if st.session_state.capturing:
        st.session_state.lenswise.capture_img(200)
    
        
    
    

if __name__ == "__main__":
    main()