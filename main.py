import os
import urllib.request
from start_ai import Doit, Train

def main():
    a = Doit('intents_dataset.json')
    a.all()
main()