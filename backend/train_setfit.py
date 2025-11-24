from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
import os

# Few-shot training data: ~5 examples per category
# This helps the model learn specific nuances (e.g., "Courtroom Drama" is Entertainment, not Law)
train_texts = [
    "The football match ended in a draw.",
    "He scored a hat-trick in the final.",
    "The Olympic games will be held next year.",
    "The tennis champion won another grand slam.",
    "The basketball team secured a last-minute victory.",
    "The government passed a new law today.",
    "The election results were announced last night.",
    "The president gave a speech at the summit.",
    "Lawmakers debated the new policy for hours.",
    "The congress approved the budget proposal.",
    "The new smartphone features a better camera.",
    "Artificial intelligence is transforming many industries.",
    "The software update improved system performance.",
    "Cloud computing enables scalable infrastructure.",
    "The startup is building a new mobile app.",
    "The movie received excellent reviews from critics.",
    "The singer released a new album.",
    "The TV series finale shocked many fans.",
    "The actor won an award for best performance.",
    "The concert attracted thousands of people.",
    "The company reported record profits this quarter.",
    "The merger created the largest bank in the country.",
    "The startup raised funding from investors.",
    "The CEO announced a new business strategy.",
    "The board approved the acquisition deal.",
    "The patient needs surgery immediately.",
    "Regular exercise improves cardiovascular health.",
    "The doctor prescribed a new medication.",
    "Vaccination helps prevent infectious diseases.",
    "A balanced diet is important for good health.",
    "The contract is valid for five years.",
    "The defendant pleaded not guilty in court.",
    "The lawyer prepared documents for the trial.",
    "The judge issued the final verdict.",
    "The agreement was signed by both parties.",
    "The stock market crashed today.",
    "Inflation rates are rising globally.",
    "He invested his savings in mutual funds.",
    "The bank increased interest rates.",
    "The company issued new corporate bonds.",
    "The researchers published a paper on quantum physics.",
    "The telescope captured images of a distant galaxy.",
    "The lab experiment yielded unexpected results.",
    "Scientists discovered a new species of bacteria.",
    "The study explores the effects of climate change.",
    "The university offers a new degree program.",
    "Students are preparing for their final exams.",
    "The teacher assigned homework for the weekend.",
    "Online learning platforms are becoming popular.",
    "The school board approved the new curriculum.",
    "The forest fire destroyed acres of land.",
    "Pollution levels in the city have decreased.",
    "Renewable energy sources are being adopted.",
    "The conservation project aims to protect wildlife.",
    "Climate change is a global crisis.",
    "The flight to Paris was delayed by two hours.",
    "We booked a hotel room with a sea view.",
    "The tour guide explained the history of the castle.",
    "Travel restrictions have been lifted.",
    "The backpacker explored the remote village.",
    "The restaurant serves authentic Italian cuisine.",
    "The recipe requires fresh ingredients.",
    "The chef prepared a delicious three-course meal.",
    "Food delivery services are very convenient.",
    "The bakery sells fresh bread every morning.",
]

train_labels = [
    "sports", "sports", "sports", "sports", "sports",
    "politics", "politics", "politics", "politics", "politics",
    "technology", "technology", "technology", "technology", "technology",
    "entertainment", "entertainment", "entertainment", "entertainment", "entertainment",
    "business", "business", "business", "business", "business",
    "health", "health", "health", "health", "health",
    "law", "law", "law", "law", "law",
    "finance", "finance", "finance", "finance", "finance",
    "science", "science", "science", "science", "science",
    "education", "education", "education", "education", "education",
    "environment", "environment", "environment", "environment", "environment",
    "travel", "travel", "travel", "travel", "travel",
    "food", "food", "food", "food", "food",
]

def main():
    print("Preparing dataset...")
    dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})

    unique_labels = sorted(list(set(train_labels)))
    label2id = {label: i for i, label in enumerate(unique_labels)}

    dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})

    print("Loading base model (sentence-transformers/paraphrase-MiniLM-L6-v2)...")
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        labels=unique_labels,
    )

    print("Starting training...")
    trainer = SetFitTrainer(
        model=model,
        train_dataset=dataset,
        metric="accuracy",
        batch_size=16,
        num_epochs=1,
        num_iterations=20,
    )

    trainer.train()

    output_dir = "setfit-model"
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    print(f"SetFit model saved to '{output_dir}'")

if __name__ == "__main__":
    main()
