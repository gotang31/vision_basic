from bs4 import BeautifulSoup
from requests import get, Session
import re
from urllib.request import urlretrieve
import os
import pickle

def prod_link(url, headers = None):
    links = []
    resp = get(url, headers = headers)
    dom = BeautifulSoup(resp.text, 'html5lib')
    for elem in dom.select('a[class="baby-product-link"]'):
        links.append(elem.attrs['href'])
    return links

def clean_filename(title):
    # 파일 이름에서 특수 문자와 공백을 제거하는 함수
    cleaned_title = re.sub(r'[\/:*?"<>|\s]', '', title)
    return cleaned_title

def img_url(url, headers):
    sess = Session()
    resp = sess.get(url, headers=headers, verify=False)
    dom = BeautifulSoup(resp.text, 'html5lib')
    matches = list(set(re.findall(r'"origin":"(//[^"]+)"', str(dom))))
    
    return matches

def img_download(links, headers, fdir): # links = product links by 'prod_link' function
    if not os.path.exists(fdir):
        # 디렉토리가 존재하지 않으면 새로 생성
        os.makedirs(fdir)
        
    for link in links:
        link = 'https://www.coupang.com' + link
        img_link = img_url(link, headers = headers)
        for link_ in img_link:
            title = re.findall('/([^/]+)$', link_)[0]
            title = clean_filename(title)
            file_path = os.path.join(fdir, title)
            urlretrieve("https:" + link_, file_path)
        print('check link:', link)

if __name__ == "__main__":

    headers = {'User-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36' , 
           'Accept-Language' : 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'}
    
    link_res = []
    for page in range(1, 18):
        url = f'https://www.coupang.com/np/categories/497135?page={page}'
        link_res.extend(prod_link(url, headers))

    
    with open('laptop_coupang.pickle', 'wb') as f:
        pickle.dump(link_res, f)
    
    img_download(link_res, headers, 'laptop')