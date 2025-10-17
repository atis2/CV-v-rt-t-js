import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from typing import List


from config import GEMINI_API_KEY 


class CVAssessment(BaseModel):
    """Schema for the CV assessment output."""
    match_score: int = Field(..., ge=0, le=100, description="Atbilstības rādītājs no 0 līdz 100.")
    summary: str = Field(..., description="Īss apraksts, cik labi CV atbilst JD.")
    strengths: List[str] = Field(..., description="Galvenās prasmes/pieredze no CV, kas atbilst JD.")
    missing_requirements: List[str] = Field(..., description="Svarīgas JD prasības, kas CV nav redzamas.")
    verdict: str = Field(..., pattern="^(strong match|possible match|not a match)$", description="Kandidāta ieteikums: 'strong match', 'possible match' vai 'not a match'.")


def read_file_content(filepath: str) -> str:
    """Nolasa faila saturu."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Kļūda: Fails nav atrasts: {filepath}")
        exit()
    except Exception as e:
        print(f"Kļūda lasot failu {filepath}: {e}")
        exit()

def create_gemini_prompt(jd_text: str, cv_text: str) -> str:
    """Ģenerē galveno promptu modelim un saglabā to prompt.md (Solī 2)."""
    
    prompt_template = f"""
Tu esi profesionāls HR speciālists un darbinieku atlases speciālists. Tavs uzdevums ir novērtēt kandidāta CV atbilstību dotajam Darba Aprakstam (JD).

Analizē sekojošos tekstus un izveido atbildi obligāti norādītajā JSON formātā.

Noteikumi:
1. Atbildes temperatūrai jābūt <= 0.3.
2. Atbildei jābūt tikai un vienīgi derīgam JSON, kas atbilst norādītajai shēmai.
3. Atbildes valodai jābūt latviešu (izņemot `verdict` lauku).
4. `match_score` ir kvantitatīvs novērtējums (0-100), kur 100 ir perfekta atbilstība.
5. `verdict` ir viens no trim vārdiem: "strong match", "possible match", vai "not a match".

DARBA APRAKSTS (JD):
---
{jd_text}
---

KANDIDĀTA CV:
---
{cv_text}
---
"""
    
    prompt_filepath = "prompt.md"
    try:
        with open(prompt_filepath, 'w', encoding='utf-8') as f:
            f.write(prompt_template)
        print(f"Prompt saglabāts: {prompt_filepath}")
    except Exception as e:
        print(f"Brīdinājums: Nevar saglabāt prompt.md: {e}")

    return prompt_template

def generate_markdown_report(cv_num: int, json_data: dict) -> str:
    """Ģenerē īsu Markdown pārskatu (solis 5)."""
    
    
    strengths_list = "\n".join([f"* {s}" for s in json_data.get('strengths', [])]) if json_data.get('strengths') else "* Nav specifisku stipro pušu, kas atbilstu JD."
    missing_list = "\n".join([f"* {m}" for m in json_data.get('missing_requirements', [])]) if json_data.get('missing_requirements') else "* Visas galvenās prasības šķiet izpildītas."

    markdown_report = f"""# Kandidāta {cv_num} CV Vērtējums

## Novērtējuma Kopsavilkums
| Kategorija | Vērtība |
| :--- | :--- |
| **Atbilstības Rādītājs** | **{json_data.get('match_score', 'N/A')}/100** |
| **Verdikts (Ieteikums)** | **{json_data.get('verdict', 'N/A').upper()}** |
| **Kopsavilkums** | {json_data.get('summary', 'Nav kopsavilkuma.')} |

---

## Galvenās Stiprās Puses (Atbilst JD)
{strengths_list}

---

## Trūkstošās Prasības (Nav CV)
{missing_list}

"""
    
    report_filepath = f"outputs/cv{cv_num}_report.md"
    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    print(f"Pārskats saglabāts: {report_filepath}")
    return markdown_report




def run_cv_assessor():
    """Izpilda visus projekta soļus."""
    print("Sākas AI darbināts CV vērtētājs (Gemini Flash 2.5 + Python)...")
    
    
    jd_text = read_file_content("sample_inputs/jd.txt")
    
    cv_files = ["sample_inputs/cv1.txt", "sample_inputs/cv2.txt", "sample_inputs/cv3.txt"]
    
    
    for i, cv_path in enumerate(cv_files):
        cv_num = i + 1
        print(f"\n--- Apstrādā Kandidātu {cv_num} ---")
        
        
        cv_text = read_file_content(cv_path)
        
        
        full_prompt = create_gemini_prompt(jd_text, cv_text)

        try:
            
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            
            config = types.GenerateContentConfig(
                temperature=0.3, 
                response_mime_type="application/json",
                response_schema=CVAssessment,
            )

            print("Izsauc Gemini Flash 2.5...")
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=full_prompt,
                config=config,
            )
            
            
            try:
                
                response_json_str = response.text.strip()
                model_output_data = json.loads(response_json_str)
            except json.JSONDecodeError as e:
                print(f"Kļūda: Nevar parsēt Gemini atbildi kā JSON. Kļūda: {e}")
                print(f"Saņemtā atbilde: {response.text}")
                continue

            
            json_filepath = f"outputs/cv{cv_num}.json"
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(model_output_data, f, indent=4, ensure_ascii=False)
            print(f"JSON izvade saglabāta: {json_filepath}")
            
            
            generate_markdown_report(cv_num, model_output_data)

        except Exception as e:
            
            print(f"Kļūda apstrādājot Kandidātu {cv_num}: {e}")

    print("\nProjekts pabeigts. Vērtējumi un pārskati atrodami 'outputs/' direktorijā.")


os.makedirs('outputs', exist_ok=True)
os.makedirs('sample_inputs', exist_ok=True) 


if __name__ == "__main__":
    
    for file_name in ["jd.txt", "cv1.txt", "cv2.txt", "cv3.txt"]:
        path = f"sample_inputs/{file_name}"
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                if "jd" in file_name:
                    f.write("Darba apraksts: Meklējam pieredzējušu Python programmētāju ar zināšanām par datubāzēm (SQL, NoSQL) un mākoņpakalpojumiem (AWS/Azure/GCP). Nepieciešama vismaz 3 gadu pieredze.")
                else:
                    f.write(f"Kandidāta {file_name.strip('cv').strip('.txt')} CV teksts.")
            print(f"Izveidots tukšs parauga fails: {path}")

    run_cv_assessor()