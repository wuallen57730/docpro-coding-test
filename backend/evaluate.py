import time
import torch
import pandas as pd
import numpy as np
from setfit import SetFitModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "setfit-model")

def load_model():
    """Load the SetFit classification model."""
    logger.info(f"Loading model from {MODEL_PATH}...")
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"SetFit model not found at {MODEL_PATH}. Please run train_setfit.py first.")
    
    model = SetFitModel.from_pretrained(MODEL_PATH)
    logger.info("Model loaded successfully.")
    return model

def evaluate_model():
    try:
        classifier = load_model()
    except Exception as e:
        logger.error(str(e))
        return

    # 1. Define Test Dataset (Golden Set)
    # In a real scenario, this would be loaded from a CSV or database
    test_data = [
        {
            "text": "This agreement shall remain in full force and effect for a period of five (5) years, commencing on the Effective Date, unless earlier terminated in accordance with the provisions set forth herein.", 
            "true_label": "law"
        },
        {
            "text": "During the arraignment hearing at the Superior Court, the defendant, represented by counsel, stood before the judge and formally entered a plea of not guilty to all charges listed in the indictment.", 
            "true_label": "law"
        },
        {
            "text": "All intellectual property rights, including but not limited to patents, copyrights, trademarks, and trade secrets, are strictly protected under international law and applicable treaties, prohibiting unauthorized reproduction or distribution.", 
            "true_label": "law"
        },
        {
            "text": "Major indices, including the S&P 500 and NASDAQ, plummeted significantly shortly after the opening bell, signaling a severe market correction driven by investor fears of an impending recession.", 
            "true_label": "finance"
        },
        {
            "text": "Central banks around the world are grappling with persistently high inflation rates, driven by supply chain disruptions and rising energy costs, forcing policymakers to consider aggressive interest rate hikes.", 
            "true_label": "finance"
        },
        {
            "text": "Due to the severity of the internal hemorrhaging and the risk of multiple organ failure, the attending physician has determined that the patient requires immediate surgical intervention to stabilize their condition.", 
            "true_label": "health"
        },
        {
            "text": "Engaging in consistent aerobic physical activity not only strengthens the heart muscle and improves blood circulation but also significantly lowers the risk of developing chronic cardiovascular diseases and hypertension.", 
            "true_label": "health"
        },
        {
            "text": "The latest flagship smartphone model boasts a revolutionary quad-lens camera system with a 108MP primary sensor, advanced optical image stabilization, and AI-powered low-light processing capabilities.", 
            "true_label": "technology"
        },
        {
            "text": "The rapid advancement of artificial intelligence, particularly in the fields of generative models and deep neural networks, is fundamentally reshaping traditional workflows and automating complex tasks across various global industries.", 
            "true_label": "technology"
        },
        {
            "text": "Despite an intense ninety minutes of play and several near-miss opportunities in stoppage time, the highly anticipated football match between the two historic rivals concluded in a goalless draw, leaving fans on both sides disappointed.", 
            "true_label": "sports"
        },
        {
            "text": "The International Olympic Committee has officially confirmed that the upcoming Olympic Games are scheduled to commence next summer, welcoming elite athletes from over 200 nations to compete for gold on the world stage.", 
            "true_label": "sports"
        },
        {
            "text": "Walking into The Saffron & Sage, you are immediately greeted by the warm, earthy aroma of roasted spices and freshly baked focaccia. The interior strikes a balance between rustic charm and modern minimalism, featuring exposed brick walls adorned with hanging copper pots. However, the acoustics were a bit problematic; as the evening progressed, the dining hall became incredibly noisy, making conversation somewhat difficult. Service was friendly but slightly scattered—our water glasses remained empty for long stretches despite the staff rushing past our table repeatedly.", 
            "true_label": "food"
        },
        # Adversarial / Tricky examples (Expanded to reinforce the contrast)
        {
            "text": "Although he spent his days litigating corporate cases, the senior lawyer dedicated his evenings to analyzing his personal portfolio, heavily investing his savings into volatile tech stocks and diversifying his assets to hedge against market inflation.", 
            "true_label": "finance"
        }, # Contains 'lawyer' and 'litigating' but the core action and context are purely finance
        {
            "text": "Following a dispute over wrongful termination and breach of contract regarding patient safety protocols, the renowned surgeon filed a formal lawsuit against the hospital administration seeking significant compensatory damages.", 
            "true_label": "law"
        }, # Contains 'surgeon', 'patient', 'hospital' but the core action is a lawsuit
        {
        "text": "Amidst growing tensions over trade tariffs, the prime minister convened an emergency cabinet meeting to discuss diplomatic strategies, ultimately deciding to propose a bipartisan bill aimed at strengthening national security while maintaining open dialogue with foreign allies.",
        "true_label": "politics"
        },
        {
            "text": "The incumbent senator launched a fierce campaign focused on voter suppression issues and redistricting, rallying supporters at the town hall to demand electoral reform and greater transparency in campaign finance reporting before the upcoming midterm elections.",
            "true_label": "politics"
        },
        # --- Category: Education (教育) ---
        {
            "text": "The university administration announced a comprehensive overhaul of the undergraduate curriculum, shifting focus towards interdisciplinary studies and experiential learning to better equip students with critical thinking skills required for the modern workforce.",
            "true_label": "education"
        },
        {
            "text": "Facing budget cuts, the local school board voted to consolidate resources by implementing a hybrid learning model, which allows students to access digital textbooks and attend virtual lectures while preserving funding for essential extracurricular activities.",
            "true_label": "education"
        },

        # --- Category: Entertainment (娛樂) ---
        {
            "text": "Following a record-breaking opening weekend at the global box office, the sci-fi blockbuster received critical acclaim for its groundbreaking visual effects and compelling cinematography, securing nominations for three major Academy Awards including Best Picture.",
            "true_label": "entertainment"
        },
        {
            "text": "The chart-topping pop icon surprised fans by dropping a surprise album at midnight, featuring collaborations with underground indie artists that blend electronic dance beats with soulful acoustic melodies, instantly trending on all major streaming platforms.",
            "true_label": "entertainment"
        },

        # --- Adversarial / Tricky Examples (對抗樣本) ---
        
        # 內容涉及「法律」詞彙，但實際上是關於「娛樂」 (電影劇情)
        {
            "text": "The courtroom drama series finale drew millions of viewers as the protagonist lawyer delivered a shocking closing argument that exposed the corruption within the fictional judicial system, earning the lead actor an Emmy award.",
            "true_label": "entertainment"
        },
        
        # 內容涉及「金錢/預算」，但實際上是關於「教育」 (學校政策)
        {
            "text": "Despite the massive fiscal deficit, the college president refused to increase tuition fees, arguing that financial aid packages must be expanded to ensure that higher learning remains accessible to students from low-income families.",
            "true_label": "education"
        },
        
        # 內容涉及「科技/AI」，但實際上是關於「政治」 (監管與政策)
        {
            "text": "Legislators are debating a controversial new framework to regulate artificial intelligence, aiming to pass laws that prevent algorithmic bias and protect user privacy without stifling innovation in the tech sector.",
            "true_label": "politics"
        }
    ]
    
    # Get labels from the model
    candidate_labels = list(classifier.labels)
    
    logger.info(f"Starting evaluation on {len(test_data)} samples...")
    
    predictions = []
    true_labels = []
    latencies = []
    
    for i, item in enumerate(test_data):
        start_time = time.time()
        # SetFit predict returns the label directly
        # predict_proba returns probabilities
        
        # We use predict for the top label
        predicted_label = classifier.predict([item["text"]])[0]
        
        # For confidence, we can use predict_proba
        probs = classifier.predict_proba([item["text"]])[0]
        # Find the index of the predicted label
        label_index = candidate_labels.index(predicted_label)
        confidence = probs[label_index]
        
        end_time = time.time()
        
        predictions.append(predicted_label)
        true_labels.append(item["true_label"])
        latencies.append(end_time - start_time)
        
        logger.info(f"Sample {i+1}: True='{item['true_label']}', Pred='{predicted_label}' (Conf: {confidence:.2f})")

    # 2. Calculate Metrics
    accuracy = accuracy_score(true_labels, predictions)
    avg_latency = np.mean(latencies)
    
    logger.info("-" * 30)
    logger.info("EVALUATION RESULTS")
    logger.info("-" * 30)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Average Latency: {avg_latency:.4f} seconds/sample")
    
    # 3. Detailed Report
    report = classification_report(true_labels, predictions, labels=candidate_labels, zero_division=0)
    logger.info("\nClassification Report:\n" + report)
    
    # 4. Confusion Matrix
    cm = confusion_matrix(true_labels, predictions, labels=candidate_labels)
    cm_df = pd.DataFrame(cm, index=candidate_labels, columns=candidate_labels)
    logger.info("\nConfusion Matrix:\n" + str(cm_df))
    
    # 5. Save results to CSV for further analysis
    results_df = pd.DataFrame({
        "text": [item["text"] for item in test_data],
        "true_label": true_labels,
        "predicted_label": predictions,
        "latency": latencies
    })
    results_df.to_csv("evaluation_results.csv", index=False)
    logger.info("Detailed results saved to 'evaluation_results.csv'")

if __name__ == "__main__":
    evaluate_model()
