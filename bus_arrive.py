#NOTE pip install python-dotenv urllib3
import os
from dotenv import load_dotenv
from urllib.parse import urlencode, quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

load_dotenv(verbose=True)
STATIONINFO_APIKEY = os.getenv('STATIONINFO_APIKEY')
BUSPOS_APIKEY = os.getenv('BUSPOS_APIKEY')

def get_bus_dict(target_station):
    url = 'http://ws.bus.go.kr/api/rest/stationinfo/getStationByUid'
    query_params = '?' + urlencode({
        quote_plus('ServiceKey') : STATIONINFO_APIKEY,
        quote_plus('arsId') : target_station })

    request = Request(url + query_params)
    request.get_method = lambda: 'GET'
    response_body = urlopen(request).read()

    tree = ET.fromstring(response_body)
    bus_list = tree.find('msgBody').findall('itemList')

    bus_come = {}
    for bus in bus_list:
        #NOTE routeType: (1:공항, 2:마을, 3:간선, 4:지선,
        # 5:순환, 6:광역, 7:인천, 8:경기, 9:폐지, 0:공용)
        if (bus.find('arrmsg1').text == '곧 도착'):
            vehicle_id = bus.find('vehId1').text

            bus_come[bus.find('rtNm').text] = (
                    bus.find('routeType').text, 
                    get_bus_plain_no(vehicle_id)
                )
        
    return bus_come

def get_bus_plain_no(vehicle_id):
    url = 'http://ws.bus.go.kr/api/rest/buspos/getBusPosByVehId'
    query_params = '?' + urlencode({
        quote_plus('ServiceKey'): BUSPOS_APIKEY,
        quote_plus('vehId') : vehicle_id
    })

    request = Request(url + query_params)
    request.get_method = lambda: 'GET'
    response_body = urlopen(request).read()

    tree = ET.fromstring(response_body)
    bus_list = tree.find('msgBody').findall('itemList')
    
    for bus in bus_list:
        return bus.find('plainNo').text
