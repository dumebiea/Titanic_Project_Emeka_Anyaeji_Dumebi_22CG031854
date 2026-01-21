import requests
import os

def download_titanic():
    urls = [
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        "https://assets.datacamp.com/production/course_1639/datasets/titanic.csv",
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for url in urls:
        try:
            print(f"Attempting download from {url}...")
            response = requests.get(url, headers=headers, verify=False, timeout=10)
            response.raise_for_status()
            
            # Verify it looks like CSV
            content = response.text
            if "Pclass" in content and "Survived" in content:
                with open("titanic.csv", "w", encoding='utf-8') as f:
                    f.write(content)
                print(f"Download successful from {url}.")
                return True
            else:
                print(f"Content from {url} did not look like Titanic dataset.")
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
    
    return False

if __name__ == "__main__":
    if not download_titanic():
        print("All downloads failed. Falling back to dummy data.")
        with open("titanic.csv", "w") as f:
            f.write("PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n")
            f.write('1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S\n')
            f.write('2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C\n')
            f.write('3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S\n')
            f.write('4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S\n')
            f.write('5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S\n')
            f.write('6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q\n')
