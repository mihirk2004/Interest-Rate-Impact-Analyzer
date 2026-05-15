import streamlit as st
from streamlit_option_menu import option_menu
import base64
from PIL import Image
import io
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="My Streamlit App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Create a CSS file if it doesn't exist
css_content = """
/* Navbar styling */
.css-1vq4p4l {
    padding: 1rem;
    background-color: #2c3e50;
    color: white;
}

/* Card styling */
.card {
    width:275px;
    height:200px;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 20px;
    background-color: white;
}

.card:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}

.card-img {
    width: 100%;
    height: 150px;
    object-fit: cover;
    border-radius: 5px 5px 0 0;
}

.card-title {
    font-size: 20px;
    font-weight: bold;
    margin: 10px 0;
}

.card-text {
    font-size: 17px;
    color: #555;
}

/* Team section styling */
.team-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 30px;
    margin: 30px 0;
}

.team-member {
    text-align: center;
}

.profile-img {
    width: 200px;
    height: 200px;
    border-radius: 50%;
    object-fit: cover;
    border: 5px solid #f1f1f1;
}

.member-name {
    font-weight: bold;
    margin-top: 10px;
}

.member-role {
    color: #666;
    font-size: 14px;
}

/* Footer styling */
.footer {
    background-color: #2c3e50;
    color: white;
    text-align: center;
    padding: 20px;
    margin-top: 50px;
}
"""

with open("style.css", "w") as f:
    f.write(css_content)

local_css("style.css")

# Sample images (base64 encoded placeholder images)
# def img_to_bytes(img_path):
#     img = Image.open(img_path)
#     buffer = io.BytesIO()
#     img.save(buffer, format="JPG")
#     return base64.b64encode(buffer.getvalue()).decode()

def image_to_base64(image_path):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")  
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_base64}"

# Placeholder images (replace with your own)
placeholder_img = "https://via.placeholder.com/300x150.png?text=Sample+Image"
mihirImg = image_to_base64("images/Mihir Image.jpg")
udayImg = image_to_base64("images/Uday.jpg")
shuImg = image_to_base64("images/Shubham.jpg")
sahilImg = image_to_base64("images/Sahil.jpg")

# Navbar
with st.container():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Services", "About", "Contact"],
        icons=["house", "list-task", "people", "telephone"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#2c3e50"},
            "icon": {"color": "orange", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#42586e",
                "color":"white"
            },
            "nav-link-selected": {"background-color": "#1abc9c"},
        }
    )

# Main content
if selected == "Home":
    st.title("Welcome to Our Home Page")
    st.markdown(
                f"""
                <h4>
                    We are providing a such important insights to a Finance Orientend Users to visualize and get future predicted different financial rates and other features. In this project we had used different Machine Learning models like XGBoost,LSTM,Linear Regression
                </h4>
                <h4>
                    So Let's Study
                </h4>
                """,
                unsafe_allow_html=True
            )
    
    # Create 8 cards in a grid
    cols = st.columns(4)
    
    card_data = [
        {
            "id":1,
            "title": "GDP and Economic Growth", 
            "desc": "This model predicts Economic Growth by considering GDP feature.", 
            "img": placeholder_img, 
            "link": "page1.py"
        },
        {
            "id":2,
            "title": "Inflation Rate and Economic Growth", 
            "desc": "This model predicts Economic Growth by considering Inflation Rate feature", 
            "img": placeholder_img, 
            "link": "page2.py"
        },
        {
            "id":3,
            "title": "Inflation Rate and Loan Rate", 
            "desc": "This model predicts Loan Rate by considering Inflation Rate feature", 
            "img": placeholder_img, 
            "link": "page3.py"
        },
        {
            "id":4,
            "title": "Interest Rate and Economic Growth", 
            "desc": "This model predicts Economic Growth by considering Interest Rate feature", 
            "img": placeholder_img, 
            "link": "page4.py"
        },
        {
            "id":5,
            "title": "Interest Rate and GDP", 
            "desc": "This model predicts GDP by considering Interest Rate feature", 
            "img": placeholder_img, 
            "link": "page5.py"
        },
        {
            "id":6,
            "title": "Interest Rate and Inflation Rate", 
            "desc": "This model predicts Inflation Rate by considering Interest Rate feature", 
            "img": placeholder_img, 
            "link": "page6.py"
        },
        {
            "id":7,
            "title": "Interest Rate and Loan Rate", 
            "desc": "This model predicts Loan Rate by considering Interest Rate feature", 
            "img": placeholder_img, 
            "link": "page7.py"
        },
        {
            "id":8,
            "title": "Interest Rate and Saving Rate", 
            "desc": "This model predicts Saving Rate by considering Interest Rate feature", 
            "img": placeholder_img, 
            "link": "page8.py"
        }
    ]
    
    for i, card in enumerate(card_data):
        with cols[i % 4]:
            with st.container():
                st.markdown(f"""
                    <div class="card">
                        <div class="card-title">{card['title']}</div>
                        <div class="card-text">{card['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
                
            if st.button("Let's Study Model", key=f"btn_{card['id']}"):
                st.switch_page(f"pages/{card['link']}")

    # # About Team section
    # st.header("Our Team")
    
    # team_members = [
    #     {"name": "Mihir Kolpuke", "role": "Model training and Fronted", "img": mihirImg},
    #     {"name": "Uday Kuthe", "role": "Model training and Dataset", "img": udayImg},
    #     {"name": "Shubham Shinde", "role": "Model training and Data Processing", "img": shuImg},
    #     {"name": "Sahil Lawande", "role": "Frontend and Documentation", "img": sahilImg},
    # ]
    
    # st.markdown('<div class="team-container">', unsafe_allow_html=True)
    
    # cols = st.columns(4)
    # for i, member in enumerate(team_members):
    #     with cols[i]:
    #         st.markdown(
    #             f"""
    #             <div class="team-member">
    #                 <img class="profile-img" src="{member['img']}" alt="{member['name']}">
    #                 <div class="member-name">{member['name']}</div>
    #                 <div class="member-role">{member['role']}</div>
    #             </div>
    #             """,
    #             unsafe_allow_html=True
    #         )
    
    # st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div class="footer">
        <p>© 2025 My Company. All rights reserved.</p>
        <p>Contact us: info@mycompany.com | +1 (123) 456-7890</p>
    </div>
    """,
    unsafe_allow_html=True
)

if selected=="Services":
    
    for i, card in enumerate(card_data):
        with cols[i % 4]:
            with st.container():
                st.markdown(f"""
                    <div class="card">
                        <div class="card-title">{card['title']}</div>
                        <div class="card-text">{card['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
                
            if st.button("Let's Study Model", key=f"btn_{card['id']}"):
                st.switch_page(f"pages/{card['link']}")
            
if selected=="Contact":
    st.markdown(
    """
    <div class="footer">
        <p>© 2025 My Company. All rights reserved.</p>
        <p>Contact us: info@mycompany.com | +1 (123) 456-7890</p>
    </div>
    """,
    unsafe_allow_html=True
    )

# if selected=="About":
#     st.markdown('<div class="team-container">', unsafe_allow_html=True)
    
#     cols = st.columns(4)
#     for i, member in enumerate(team_members):
#         with cols[i]:
#             st.markdown(
#                 f"""
#                 <div class="team-member">
#                     <img class="profile-img" src="{member['img']}" alt="{member['name']}">
#                     <div class="member-name">{member['name']}</div>
#                     <div class="member-role">{member['role']}</div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
    
    st.markdown('</div>', unsafe_allow_html=True)