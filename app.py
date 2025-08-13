from flask import Flask, render_template, request, jsonify
import json
import os
import re
import requests
import base64
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from PIL import Image
import pypdf
import docx
from datetime import datetime
from datetime import timedelta 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)

gemini_api_key = os.getenv('GEMINI_API_KEY')
serper_api_key = os.getenv('SERPER_API_KEY')
mem0_api_key = os.getenv('MEM0_API_KEY')

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=gemini_api_key
)

langchain_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    api_key=gemini_api_key
)

serper_tool = SerperDevTool(
    api_key=serper_api_key,
    n_results=50
)

class DiseaseResult(BaseModel):
    disease: str
    plant: str
    symptoms: str
    remedies: str
    resources: List[Dict[str, str]]

class Scheme(BaseModel):
    name: str
    category: str
    description: str
    link: str

class CropPlan(BaseModel):
    crop_name: str
    sowing_time: str
    cultivation_tips: str
    expected_yield: str

class SchemeFilterTool(BaseTool):
    name: str = Field(default="SchemeFilterTool", description="Filters schemes based on user criteria with priority.")
    description: str = Field(default="Filters government schemes by prioritizing location, then occupation, then caste, gender, and landholding.")

    def _run(self, schemes: List[Dict[str, Any]], user_data: dict) -> List[Dict[str, Any]]:
        try:
            filtered_schemes = []
            for scheme in schemes:
                location = user_data.get('location', '').lower()
                occupation = user_data.get('occupation', '').lower()
                caste = user_data.get('caste', '').lower()
                gender = user_data.get('gender', '').lower()
                landholding = user_data.get('landholding', '').lower()

                name_lower = scheme.get('name', '').lower()
                category_lower = scheme.get('category', '').lower()
                desc_lower = scheme.get('description', '').lower()

                location_match = (
                    location in name_lower or
                    location in desc_lower or
                    location in scheme.get('link', '').lower() or
                    'assam' in desc_lower
                )

                occupation_match = (
                    occupation in category_lower or
                    occupation in desc_lower or
                    'farmer' in desc_lower or
                    'agriculture' in category_lower or
                    'agribusiness' in category_lower
                )

                other_match = (
                    caste in desc_lower or
                    gender in desc_lower or
                    (landholding and (
                        'land' in desc_lower or
                        'small' in desc_lower or
                        'marginal' in desc_lower or
                        landholding in desc_lower
                    ))
                )

                if location_match or occupation_match or other_match:
                    filtered_schemes.append(scheme)

            filtered_schemes.sort(key=lambda x: (
                user_data.get('location', '').lower() not in x.get('description', '').lower() and
                user_data.get('location', '').lower() not in x.get('name', '').lower() and
                user_data.get('location', '').lower() not in x.get('link', '').lower(),
                user_data.get('occupation', '').lower() not in x.get('category', '').lower() and
                user_data.get('occupation', '').lower() not in x.get('description', '').lower()
            ))
            return filtered_schemes[:20]
        except Exception as e:
            logger.error(f"Error filtering schemes: {str(e)}")
            return f"Error filtering schemes: {e}"

class CropDiseaseAPI(BaseTool):
    name: str = Field(default="CropDiseaseAPI", description="Tool to detect crop diseases from image using external ML API.")
    description: str = Field(default="Identifies crop disease by sending base64 image to susya.onrender.com API.")

    def _run(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as img_file:
                imgdata = base64.b64encode(img_file.read()).decode("utf-8")
            response = requests.post("https://susya.onrender.com", json={"image": imgdata})
            response.raise_for_status()

            data = json.loads(response.text)
            disease = data.get("disease", "Unknown disease")
            plant = data.get("plant", "Unknown plant")

            return f"Disease: {disease}\nPlant: {plant}"
        except Exception as e:
            logger.error(f"Error calling Crop Disease API: {str(e)}")
            return f"Error calling Crop Disease API: {e}"
        
class SoilTypeAPI(BaseTool):
    name: str = Field(default="SoilTypeAPI", description="Tool to detect soil type based on location.")
    description: str = Field(default="Identifies soil type by querying an external soil database API based on location.")

    def _run(self, location: str) -> str:
        try:
            soil_map = {
                'assam': 'Alluvial',
                'punjab': 'Alluvial',
                'tamil nadu': 'Red',
                'maharashtra': 'Black',
                'karnataka': 'Red',
                'gujarat': 'Sandy'
            }
            location = location.lower().strip()
            for key, value in soil_map.items():
                if key in location:
                    return value
            return 'Unknown'
        except Exception as e:
            logger.error(f"Error detecting soil type: {str(e)}")
            return 'Unknown'
        
scheme_filter_tool = SchemeFilterTool()
crop_disease_tool = CropDiseaseAPI()
soil_type_tool = SoilTypeAPI()
        
symptoms_advisor = Agent(
    role="Crop Symptoms Specialist",
    goal="Identify and describe common symptoms of crop diseases.",
    backstory="An AI agronomist specializing in recognizing and describing symptoms of crop diseases for farmers.",
    verbose=True,
    llm=llm
)

remedy_advisor = Agent(
    role="Agro Remedy Consultant",
    goal="Suggest effective solutions for crop diseases.",
    backstory="An expert agronomist helping farmers treat plant infections.",
    verbose=True,
    llm=llm
)

resource_link_finder = Agent(
    role="Agro Web Researcher",
    goal="Find helpful guides and links about crop disease treatments.",
    backstory="An AI assistant with access to the web for agricultural research.",
    tools=[serper_tool],
    verbose=True,
    llm=llm
)

scheme_researcher = Agent(
    role="Scheme Researcher",
    goal="Search and filter government agricultural schemes based on user criteria, prioritizing location, then occupation, then other criteria.",
    backstory="An expert in finding and filtering relevant government schemes for farmers using web search tools.",
    tools=[serper_tool, scheme_filter_tool],
    verbose=True,
    llm=llm
)

crop_planner = Agent(
    role="Crop Planning Expert",
    goal="Generate personalized crop recommendations based on location, season, soil type, and land size.",
    backstory="An AI agronomist specializing in crop selection and cultivation advice for Indian farmers.",
    tools=[serper_tool, soil_type_tool],
    verbose=True,
    llm=llm
)



@app.route('/')
def index():
    return render_template('home.html')

@app.route('/api/get_soil_type', methods=['POST'])
def get_soil_type():
    try:
        data = request.get_json()
        if not data or 'location' not in data:
            return jsonify({"error": "Location is required"}), 400

        location = data['location'].strip()
        if not location:
            return jsonify({"error": "Location cannot be empty"}), 400

        soil_type = soil_type_tool._run(location)
        return jsonify({"soil_type": soil_type}), 200
    except Exception as e:
        logger.error(f"Error in get_soil_type: {str(e)}")
        return jsonify({"error": f"Error detecting soil type: {str(e)}"}), 500

@app.route('/crop_planning', methods=['GET', 'POST'])
def crop_planning():
    error_message = None
    plans = []
    form_submitted = False
    output_file = 'crop_plans.json'

    if request.method == 'POST':
        try:
            user_data = {
                'location': request.form.get('location', ''),
                'season': request.form.get('season', ''),
                'soil_type': request.form.get('soil_type', ''),
                'land_size': request.form.get('land_size', '')
            }

            required_fields = ['location', 'season', 'soil_type', 'land_size']
            if not all(user_data[field] for field in required_fields):
                error_message = "All fields are required."
                return render_template('crop_planning.html', error_message=error_message, plans=plans, form_submitted=True)

            search_task = Task(
                description=(
                    f"Search for suitable crops for a farmer in {user_data['location']} during the {user_data['season']} season, "
                    f"with {user_data['soil_type']} soil and {user_data['land_size']} acres of land. "
                    f"Use the SerperDevTool to find relevant agricultural data for India. "
                    f"Return a JSON list of 3-5 crop recommendations with crop_name, sowing_time, cultivation_tips, and expected_yield, "
                    f"ensuring the output is a valid JSON string without markdown code fences, "
                    f"compatible with the following schema: "
                    f"{json.dumps([{'crop_name': 'string', 'sowing_time': 'string', 'cultivation_tips': 'string', 'expected_yield': 'string'}])}"
                ),
                expected_output="A JSON list of 3-5 crop recommendations with crop_name, sowing_time, cultivation_tips, and expected_yield.",
                agent=crop_planner,
                output_file=output_file
            )

            crew = Crew(
                agents=[crop_planner],
                tasks=[search_task],
                verbose=True
            )

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type(Exception),
                after=lambda retry_state: logger.debug(f"Retry attempt {retry_state.attempt_number} failed with {retry_state.outcome.exception()}")
            )
            def execute_crew():
                return crew.kickoff()

            try:
                result = execute_crew()
            except Exception as e:
                logger.error(f"Crew execution failed after retries: {str(e)}")
                error_message = "The crop planning service is temporarily unavailable. Please try again later."
                return render_template('crop_planning.html', error_message=error_message, plans=plans, form_submitted=True)

            try:
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    raw_output_file = 'crop_plans_raw.txt'
                    with open(raw_output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.debug(f"Raw output saved to {raw_output_file}: {content}")

                    clean_content = content
                    if content.startswith('```json') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()
                    elif content.startswith('```') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()

                    if not clean_content:
                        error_message = "Output file is empty after cleaning."
                        logger.error(error_message)
                        return render_template('crop_planning.html', error_message=error_message, plans=plans, form_submitted=True)

                    try:
                        plans_data = json.loads(clean_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON from {output_file}: {e}")
                        error_message = f"Error processing crop plans: {e}"
                        return render_template('crop_planning.html', error_message=error_message, plans=plans, form_submitted=True)

                    if not isinstance(plans_data, list):
                        error_message = "Crop plans data is not a valid JSON list."
                        logger.error(error_message)
                        return render_template('crop_planning.html', error_message=error_message, plans=plans, form_submitted=True)

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(plans_data, f, indent=2)
                    logger.debug(f"Saved cleaned JSON to {output_file}")

                    plans = [CropPlan(**plan) for plan in plans_data if isinstance(plan, dict)]
                    if not plans:
                        error_message = "No valid crop plans found matching your criteria."
                        logger.warning(error_message)
                else:
                    error_message = f"Output file {output_file} not found."
                    logger.error(error_message)
            except Exception as e:
                logger.error(f"Error processing {output_file}: {e}")
                error_message = f"Error processing crop plans: {e}"

            form_submitted = True
        except Exception as e:
            logger.error(f"Error fetching crop plans: {e}")
            error_message = "An unexpected error occurred during crop planning. Please try again later."

    return render_template('crop_planning.html', plans=plans, error_message=error_message, form_submitted=form_submitted)

@app.route('/schemes', methods=['GET', 'POST'])
def schemes():
    error_message = None
    schemes = []
    form_submitted = False
    output_file = 'schemes_found.json'

    if request.method == 'POST':
        try:
            user_data = {
                'name': request.form.get('name', ''),
                'age': request.form.get('age', ''),
                'caste': request.form.get('caste', ''),
                'location': request.form.get('location', ''),
                'occupation': request.form.get('occupation', ''),
                'gender': request.form.get('gender', ''),
                'landholding': request.form.get('landholding', '')
            }

            required_fields = ['name', 'caste', 'location', 'occupation', 'gender', 'landholding']
            if not all(user_data[field] for field in required_fields):
                error_message = "All fields are required."
                return render_template('schemes.html', error_message=error_message, schemes=schemes, form_submitted=True)

            search_task = Task(
                description=(
                    f"Search for government agricultural schemes in India relevant to a {user_data['occupation']} "
                    f"with caste {user_data['caste']}, gender {user_data['gender']}, located in {user_data['location']}, "
                    f"and owning {user_data['landholding']} acres of land. Use the SchemeFilterTool to filter schemes "
                    f"by prioritizing location, then occupation, then caste, gender, and landholding. "
                    f"Return a JSON list of up to 50 schemes with name, category, description, and official link, "
                    f"ensuring the output is a valid JSON string without markdown code fences, "
                    f"compatible with the following schema: "
                    f"{json.dumps([{'name': 'string', 'category': 'string', 'description': 'string', 'link': 'string'}])}"
                ),
                expected_output="A JSON list of up to 50 schemes with name, category, description, and link.",
                agent=scheme_researcher,
                output_file=output_file
            )

            crew = Crew(
                agents=[scheme_researcher],
                tasks=[search_task],
                verbose=True
            )

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type(Exception),
                after=lambda retry_state: logger.debug(f"Retry attempt {retry_state.attempt_number} failed with {retry_state.outcome.exception()}")
            )
            def execute_crew():
                return crew.kickoff()

            try:
                result = execute_crew()
            except Exception as e:
                logger.error(f"Crew execution failed after retries: {str(e)}")
                error_message = "The scheme search service is temporarily unavailable. Please try again later."
                return render_template('schemes.html', error_message=error_message, schemes=schemes, form_submitted=True)

            try:
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    raw_output_file = 'schemes_found_raw.txt'
                    with open(raw_output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.debug(f"Raw output saved to {raw_output_file}: {content}")

                    clean_content = content
                    if content.startswith('```json') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()
                    elif content.startswith('```') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()

                    if not clean_content:
                        error_message = "Output file is empty after cleaning."
                        logger.error(error_message)
                        return render_template('schemes.html', error_message=error_message, schemes=schemes, form_submitted=True)

                    try:
                        schemes_data = json.loads(clean_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON from {output_file}: {e}")
                        error_message = f"Error processing scheme data: {e}"
                        return render_template('schemes.html', error_message=error_message, schemes=schemes, form_submitted=True)

                    if not isinstance(schemes_data, list):
                        error_message = "Schemes data is not a valid JSON list."
                        logger.error(error_message)
                        return render_template('schemes.html', error_message=error_message, schemes=schemes, form_submitted=True)

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(schemes_data, f, indent=2)
                    logger.debug(f"Saved cleaned JSON to {output_file}")

                    schemes = [Scheme(**scheme) for scheme in schemes_data if isinstance(scheme, dict)]
                    if not schemes:
                        error_message = "No valid schemes found matching your criteria."
                        logger.warning(error_message)
                else:
                    error_message = f"Output file {output_file} not found."
                    logger.error(error_message)
            except Exception as e:
                logger.error(f"Error processing {output_file}: {e}")
                error_message = f"Error processing scheme data: {e}"

            form_submitted = True
        except Exception as e:
            logger.error(f"Error fetching scheme data: {e}")
            error_message = "An unexpected error occurred during scheme search. Please try again later."

    return render_template('schemes.html', schemes=schemes, error_message=error_message, form_submitted=form_submitted)


@app.route('/disease', methods=['GET', 'POST'])
def disease():
    error_message = None
    result = None
    form_submitted = False
    output_file = 'disease_results.json'
    raw_output_file = 'disease_results_raw.txt'

    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                error_message = "No image file uploaded."
                return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

            image = request.files['image']
            if image.filename == '':
                error_message = "No image file selected."
                return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

            image_path = os.path.join('uploads', image.filename)
            os.makedirs('uploads', exist_ok=True)
            image.save(image_path)

            try:
                with open(image_path, "rb") as img_file:
                    imgdata = base64.b64encode(img_file.read()).decode("utf-8")
                response = requests.post("https://susya.onrender.com", json={"image": imgdata})
                response.raise_for_status()

                disease_data = json.loads(response.text)
                disease = disease_data.get("disease", "Unknown disease")
                plant = disease_data.get("plant", "Unknown plant")
            except Exception as e:
                logger.error(f"Error calling Crop Disease API: {str(e)}")
                error_message = f"Error calling Crop Disease API: {str(e)}"
                return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

            symptoms_task = Task(
                description=f"For the identified crop disease '{disease}' affecting '{plant}', provide a single concise sentence (15-25 words) describing the most prominent symptoms (e.g., leaf spots, wilting, discoloration). Example: 'Leaves show yellowing with black spots and wilting stems.'",
                expected_output="A single sentence describing symptoms (15-25 words).",
                agent=symptoms_advisor
            )

            remedies_task = Task(
                description=f"For the identified crop disease '{disease}' affecting '{plant}', provide exactly 10 concise remedy points (covering natural, chemical, and prevention advice) in 80-100 words total. Return a single string with period-separated sentences, suitable for splitting into list items. Example: 'Prune infected leaves. Apply neem oil weekly. Use chlorothalonil every 7 days. Rotate crops annually. Ensure proper drainage. Remove plant debris. Apply baking soda spray. Use resistant varieties. Avoid overhead watering. Monitor plants regularly.'",
                expected_output="A string of 10 period-separated sentences describing remedies (80-100 words).",
                agent=remedy_advisor
            )

            resource_links_task = Task(
                description=f"Search the internet for tutorials, guides, or PDFs on how to treat the crop disease '{disease}' affecting '{plant}'. Return a JSON list of 3-5 resources with title, link, and summary: [{{'title': 'string', 'link': 'string', 'summary': 'string'}}].",
                expected_output="A JSON list of 3-5 resources with title, link, and summary.",
                agent=resource_link_finder,
                output_file=output_file,
                output_format='json'
            )

            crew = Crew(
                agents=[symptoms_advisor, remedy_advisor, resource_link_finder],
                tasks=[symptoms_task, remedies_task, resource_links_task],
                verbose=True
            )

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type(Exception),
                after=lambda retry_state: logger.debug(f"Retry attempt {retry_state.attempt_number} failed with {retry_state.outcome.exception()}")
            )
            def execute_crew():
                return crew.kickoff()

            try:
                crew_result = execute_crew()
            except Exception as e:
                logger.error(f"Crew execution failed after retries: {str(e)}")
                error_message = "The disease detection service is temporarily unavailable. Please try again later."
                return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

            try:
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    with open(raw_output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.debug(f"Raw output saved to {raw_output_file}: {content}")

                    clean_content = content
                    if content.startswith('```json') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()
                    elif content.startswith('```') and content.endswith('```'):
                        clean_content = '\n'.join(content.splitlines()[1:-1]).strip()

                    if not clean_content:
                        error_message = "Output file is empty after cleaning."
                        logger.error(error_message)
                        return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

                    try:
                        resources_data = json.loads(clean_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON from {output_file}: {e}")
                        error_message = "Error processing disease analysis results. Please try again."
                        return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

                    if not isinstance(resources_data, list):
                        error_message = "Resources data is not a valid JSON list."
                        logger.error(error_message)
                        return render_template('disease.html', error_message=error_message, result=result, form_submitted=True)

                    symptoms_output = symptoms_task.output.raw if symptoms_task.output else "No symptoms identified."
                    remedies_output = remedies_task.output.raw if remedies_task.output else "No remedies identified."

                    for output in [symptoms_output, remedies_output]:
                        if output:
                            try:
                                data = json.loads(output)
                                if isinstance(data, dict):
                                    if output == symptoms_output:
                                        symptoms_output = data.get('symptoms', 'No symptoms identified.')
                                    else:
                                        remedies_output = (
                                            f"Natural: {data.get('natural', '')}. "
                                            f"Chemical: {data.get('chemical', '')}. "
                                            f"Prevent: {data.get('prevention', '')}."
                                        ).strip()
                            except json.JSONDecodeError:
                                pass

                            sentences = [s.strip() for s in output.split('.') if s.strip()]
                            output = '. '.join(sentences) + ('.' if sentences else '')
                            if output == symptoms_output:
                                symptoms_output = output
                            else:
                                remedies_output = output

                            words = output.split()
                            max_words = 25 if output == symptoms_output else 100
                            if len(words) > max_words:
                                output = ' '.join(words[:max_words-10]) + '...'
                                logger.debug(f"Truncated {'symptoms' if output == symptoms_output else 'remedies'} to ~{max_words-10} words: {output}")
                                if output == symptoms_output:
                                    symptoms_output = output
                                else:
                                    remedies_output = output

                    result = {
                        'disease': disease,
                        'plant': plant,
                        'symptoms': symptoms_output,
                        'remedies': remedies_output,
                        'resources': resources_data
                    }

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                    logger.debug(f"Saved cleaned JSON to {output_file}")

                    result = DiseaseResult(
                        disease=result['disease'],
                        plant=result['plant'],
                        symptoms=result['symptoms'],
                        remedies=result['remedies'],
                        resources=result['resources']
                    )
                else:
                    error_message = f"Output file {output_file} not found."
                    logger.error(error_message)
            except Exception as e:
                logger.error(f"Error processing {output_file}: {e}")
                error_message = f"Error processing disease analysis results: {str(e)}"

            form_submitted = True

            if os.path.exists(image_path):
                os.remove(image_path)

        except Exception as e:
            logger.error(f"Error processing disease detection: {str(e)}")
            error_message = "An unexpected error occurred during disease detection. Please try again later."

    return render_template('disease.html', error_message=error_message, result=result, form_submitted=form_submitted)

@app.route('/document_analyzer')
def document_analyzer():
    lang = request.args.get('lang', 'en')
    return render_template('document_analyzer.html', lang=lang)

@app.route('/analyze_document', methods=['POST'])
def analyze_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        document_type = request.form.get('document_type', 'other')
        language = request.form.get('language', 'en')

        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        # Validate file type
        allowed_extensions = {'pdf', 'jpg', 'jpeg', 'png', 'doc', 'docx'}
        if file.filename.rsplit('.', 1)[-1].lower() not in allowed_extensions:
            return jsonify({"error": "Unsupported file type. Use PDF, JPG, PNG, DOC, or DOCX."}), 400

        # Basic content extraction (simulated OCR for images/PDFs, direct for DOCX)
        content = ""
        if file.filename.endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                content += page.extract_text() or ""
        elif file.filename.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(file)
            content = f"Sample {document_type} document content extracted from image."
        elif file.filename.endswith(('.doc', '.docx')):
            doc = docx.Document(file)
            content = "\n".join([para.text for para in doc.paragraphs])

        if not content.strip():
            content = f"Placeholder content for {document_type} document."

        # Define prompt template
        prompt_template = PromptTemplate(
            input_variables=["content", "document_type", "language"],
            template=""" 
            You are a financial document analysis assistant for India. Analyze the provided document content and provide guidance for filling it out correctly. The document type is '{document_type}' and the output must be in '{language}'.

            Document Content:
            {content}

            Instructions:
            - Analyze the document content and identify its purpose and requirements.
            - Provide the following in '{language}':
              - Summary: A brief description of the document's purpose (1-2 sentences).
              - Required Information: A list of 3-5 key fields or details needed to complete the document.
              - Filing Instructions: A list of 3-5 steps to correctly fill out or submit the document.
              - Important Notes: Any additional guidance or requirements (e.g., supporting documents, mandatory fields).
            - Return a JSON object with translations for English, Hindi, and Kannada, even if the requested language is only one of them.
            - Ensure the JSON is valid and properly formatted.
            - Do not include markdown or extra text, only the JSON object.

            Example Output:
            {{
                "en": {{
                    "summary": "This is a loan application form requiring personal and financial details.",
                    "required_info": ["Name and address", "Monthly income", "Loan amount", "Purpose of loan", "Repayment period"],
                    "instructions": ["Fill in personal details in BLOCK LETTERS", "Provide accurate income", "State loan purpose clearly", "Include bank details", "Sign the form"],
                    "notes": "Attach Aadhaar/PAN, address proof, and income proof. All fields marked with * are mandatory."
                }},
                "hi": {{
                    "summary": "यह एक ऋण आवेदन पत्र है जिसमें व्यक्तिगत और वित्तीय विवरण की आवश्यकता है।",
                    "required_info": ["नाम और पता", "मासिक आय", "ऋण राशि", "ऋण का उद्देश्य", "पुनर्भुगतान अवधि"],
                    "instructions": ["व्यक्तिगत विवरण बड़े अक्षरों में भरें", "सटीक आय प्रदान करें", "ऋण का उद्देश्य स्पष्ट करें", "बैंक विवरण शामिल करें", "फॉर्म पर हस्ताक्षर करें"],
                    "notes": "आधार/पैन, पते का प्रमाण और आय प्रमाण संलग्न करें। * के साथ चिह्नित सभी फ़ील्ड अनिवार्य हैं।"
                }},
                "kn": {{
                    "summary": "ಇದು ವೈಯಕ್ತಿಕ ಮತ್ತು ಆರ್ಥಿಕ ವಿವರಗಳನ್ನು ಕೋರುವ ಸಾಲದ ಅರ್ಜಿ ನಮೂನೆಯಾಗಿದೆ.",
                    "required_info": ["ಹೆಸರು ಮತ್ತು ವಿಳಾಸ", "ಮಾಸಿಕ ಆದಾಯ", "ಸಾಲದ ಮೊತ್ತ", "ಸಾಲದ ಉದ್ದೇಶ", "ಮರುಪಾವತಿ ಅವಧಿ"],
                    "instructions": ["ವೈಯಕ್ತಿಕ ವಿವರಗಳನ್ನು ದೊಡ್ಡ ಅಕ್ಷರಗಳಲ್ಲಿ ಭರ್ತಿ ಮಾಡಿ", "ನಿಖರವಾದ ಆದಾಯವನ್ನು ಒದಗಿಸಿ", "ಸಾಲದ ಉದ್ದೇಶವನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ತಿಳಿಸಿ", "ಬ್ಯಾಂಕ್ ವಿವರಗಳನ್ನು ಸೇರಿಸಿ", "ಫಾರ್ಮ್‌ಗೆ ಸಹಿ ಮಾಡಿ"],
                    "notes": "ಆಧಾರ್/ಪ್ಯಾನ್, ವಿಳಾಸದ ಪುರಾವೆ ಮತ್ತು ಆದಾಯದ ಪುರಾವೆಯನ್ನು ಲಗತ್ತಿಸಿ. * ಗುರುತಿನ ಎಲ್ಲಾ ಕ್ಷೇತ್ರಗಳು ಕಡ್ಡಾಯವಾಗಿವೆ."
                }}
            }}
            """
        )

        prompt = prompt_template.format(
            content=content[:1000],
            document_type=document_type,
            language={'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada'}[language]
        )

        try:
            logger.debug(f"Sending prompt to Gemini: {prompt[:200]}...")
            response = langchain_llm.invoke(prompt)
            logger.debug(f"Raw Gemini response: {response.content}")

            response_content = response.content.strip()
            response_content = re.sub(r'^```json\s*|\s*```$', '', response_content).strip()
            logger.debug(f"Cleaned Gemini response: {response_content}")

            analysis = json.loads(response_content)
            if not isinstance(analysis, dict) or not all(lang in analysis for lang in ['en', 'hi', 'kn']):
                logger.error("Invalid analysis response format")
                return jsonify({"error": "Invalid analysis response"}), 500

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            return jsonify({"error": "Failed to parse analysis response"}), 500
        except Exception as e:
            logger.error(f"Gemini query error: {str(e)}")
            return jsonify({"error": f"Analysis error: {str(e)}"}), 500

        return jsonify({"analysis": analysis}), 200

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/telegram')
def telegram():
    return render_template('telegram.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)