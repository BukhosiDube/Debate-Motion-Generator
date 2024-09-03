import requests
from bs4 import BeautifulSoup
import sqlite3



#This function creates a database and table
def create_database():
    conn = sqlite3.connect('hello_motions.db') #Creates a connection to the SQLite database
    c = conn.cursor() #Creates a cursor object to interact with the database
    c.execute('''
              CREATE TABLE IF NOT EXISTS motions
              (id INTEGER PRIMARY KEY,
              title TEXT,
              category TEXT,
              date TEXT)
              ''') #Creates a table with the columns id, title, category, and date(change and update)
    conn.commit() #Commit the transaction
    conn.close() #Closes the database


#Thus function inserts the data into the database
def insert_data(date, tournament, round, motion):
    conn = sqlite3.connect('hello_motions.db')
    c = conn.cursor()
    c.execute("INSERT INTO motions (date, tournament, round, motion) VALUES (?, ?, ?, ?)",
              (date, tournament, round, motion)) #Inserts data into the motions table created above
    conn.commit()
    conn.close()

#Similar as to first method but this method inserst data with c.execute

#Function to scrape the data from the table
def scrape_hello_motions(url):
    response = requests.get(url) #Sends a GET request to the url
    if response.status_code == 200: #200 is the code so if it gets that then the request was successful
        soup = BeautifulSoup(response.content, 'html.parser') #Parse the HTML content of the webpage

        table = soup.find('table', {'id': "table table-striped"}) #Searches for specific table with data we want
        rows = table.find_all('tr') #Searches for all of the table rows in the table

        for row in rows[1:]: #Iterates over the rows, starts at 1 in order to skip the header row
            cols = row.find_all('td') #find the table cells
            if len(cols) == 4: #Checks if the row has exactly 4 columns
                date = cols[0].text.strip()
                tournament = cols[1].text.strip()
                round = cols[2].text.strip()
                motion = cols[3].text.strip()
                #Extracts and strips the text from each column and stores in the variable
                insert_data(date, tournament, round, motion) # Data is then inserted into a row in the database
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")


if __name__ == "__main__":
    create_database()
    scrape_hello_motions('https://hellomotions.com/search?q=Dan+Lahav%2C+Mubarrat+Wassey%2C+Bobbi+Leet%2C+Boemo+Delano+Phirinyane%2C+Connor+O%27Brien%2C+Milos+Marjanovic%2C+Sebastian+Dasso%2C+Sooyoung+Park%2C+Teck+Wei+Tan%2C+Tejal+Patwardhan&intl=0')
    #Just runs the methods above so the whole script runs