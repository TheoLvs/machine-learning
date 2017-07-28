#!/usr/bin/env python
# -*- coding: utf-8 -*-

from twilio.rest import TwilioRestClient
from slackclient import SlackClient
from urllib2 import Request, urlopen, URLError
import json

class AI():
    '''---------------------------------------------------------------------------------------------------------------'''
    '''INITIALIZATION'''
    def __init__(self,username = "Unknown",num = "xx",hometown_id = '2988507'):
        '''CONFIGURATION'''
        self.config = {}
        
        #Twilio configuration
        self.config['twilio'] = {}
        self.config['twilio']['id'] = "xx"
        self.config['twilio']['token'] = "xx"
        self.config['twilio']['from_num'] = "xx"
        
        #Open Weather configuration
        self.config['openweather'] = {}
        self.config['openweather']['id'] = 'xx'
        
        #Slack configuration
        self.config['slack'] = {}
        self.config['slack']['token'] = "xx"
        
        '''USER PROFILE'''
        self.user_profile = {}
        self.user_profile['username'] = username
        self.user_profile['num'] = num
        self.user_profile['hometown_id'] = hometown_id
        
        '''AI PROFILE'''
        self.name = "AI"
        
        
    '''---------------------------------------------------------------------------------------------------------------'''
    '''COMMUNICATION METHODS'''
    def send_SMS(self,body = "Hello there !",to_num = ""):
        if to_num == "":
            to_num = self.user_profile['num']
        client = TwilioRestClient(self.config['twilio']['id'], self.config['twilio']['token'])
        message = client.messages.create(to = to_num,from_ = self.config['twilio']['from_num'],body = body)
        
    def send_slack(self,body = "Hello there",channel = "#general"):
        sc = SlackClient(self.config['slack']['token'])
        sc.api_call("chat.postMessage", channel=channel, text=body,username=self.name, icon_emoji=':monkey:') #icon_emoji=':robot_face:'

    
    '''---------------------------------------------------------------------------------------------------------------'''
    '''INFORMATION METHODS'''
    def weather_tracker(self,n = 3):
        request = Request('http://api.openweathermap.org/data/2.5/forecast/city?id={0}&APPID={1}&units=metric'.format(
                self.user_profile['hometown_id'],
                self.config['openweather']['id']))
        try:
            response = json.loads(urlopen(request).read())
            info = {}
            info['city'] = str(response['city']['name'])
            info['list'] = [{'time':x['dt_txt'],
                             'description':x['weather'][0]['description'],
                             'temperature':x['main']['temp']}
                             for x in response['list'][:n]]
            info['list'] = [Weather_snapshot(**x) for x in info['list']]
            #print(info)
            return info
        except URLError, e:
            print("Error :",e)
    
    def weather_report(self,slack = False):
        weather = self.weather_tracker(n = 10)
        weather_list = weather['list']
        now = weather_list[0]
        after = weather_list[1:]

        '''SAYING HELLO'''
        if now.period in ['dawn','morning']:
            hello = "Good morning"
        elif now.period in ['noon']:
            hello = "Bon appétit"
        elif now.period in ['afternoon']:
            hello = "Good afternoon"
        elif now.period in ['evening']:
            hello = "Good evening"
        else:
            hello = "Good night"
        hello += " {0} !".format(self.user_profile['username'])
        report = ["------------------------",hello]
        
        '''CITY'''
        date = "/".join(map(str,[now.day,now.month,now.year]))
        report += ["Report in {0} on the {1}.".format(weather['city'],date)]
        
        '''NOW'''
        report += ['It is {0}:00'.format(now.hour)]
        if slack:
            report += ["The weather is {0} {1}".format(now.mood,now.slack_emoji)]
        else:
            report += ["The weather is {0}".format(now.mood)]
        report += ['The temperature is {0}°'.format(now.temperature)]
        if now.hour >= 14 or now.hour <= 4:
            tomorrow = min(after,key = lambda x:(abs(x.hour-12)))
            if now.level != tomorrow.level:
                change = "better" if now.level < tomorrow.level else "less good"
                report += ["Tomorrow, the weather will be {0}".format(change)]
                report += ["It will be {0} and the temperature {1}°".format(tomorrow.mood,tomorrow.temperature)]
            else:
                if slack:
                    report += ["Tomorrow the weather will be {0} and the temperature {1}° {2}".format(tomorrow.mood,tomorrow.temperature,tomorrow.slack_emoji)]
                else:
                    report += ["Tomorrow the weather will be {0} and the temperature {1}°".format(tomorrow.mood,tomorrow.temperature)]
            
        return "\n".join(report)
    
    def send_weather(self,methods = ["SMS"]):
        print("Sending weather report ...")
        for method in methods:
            if method == "SMS":
                self.send_SMS(self.weather_report())
            elif method == "slack":
                self.send_slack(self.weather_report(slack = True),channel = "#weather")
        print('Sending weather report OK"')

class Weather_snapshot():
    def __init__(self,time,description,temperature):
        '''BASIC CONFIGURATION'''
        self.time = time
        self.description = description
        self.temperature = int(temperature)
        
        '''TIME DATA'''
        self.day = int(self.time[8:10])
        self.month = int(self.time[5:7])
        self.year = int(self.time[0:4])
        self.hour = int(self.time[11:13])
        
        #Seasons
        if self.month <= 2:
            self.season = "winter"
        elif self.month < 6:
            self.season = "spring"
        elif self.month < 10:
            self.season = "summer"
        else:
            self.season = "autumn"
            
        #Period of the day
        if (self.hour >= 4 and self.hour <= 7):
            self.period = "dawn"
        elif (self.hour > 7 and self.hour <= 11):
            self.period = "morning"
        elif (self.hour > 11 and self.hour <= 14):
            self.period = "noon"
        elif (self.hour > 14 and self.hour <= 18):
            self.period = "afternoon"
        elif (self.hour > 18 and self.hour <= 21):
            self.period = "evening"
        else:
            self.period = "night"
            
        '''WEATHER MOOD LEVELS'''
        
        if self.intersection({'snow'}):
            self.level = 1
        elif self.intersection({'rain'}):
            self.level = 2
        elif self.intersection({'cloudy','clouds','cloud'}):
            self.level = 3
        elif self.intersection({'clear'}):
            self.level = 4
        elif self.intersection({'sun','sunny'}):
            self.level = 5
        else:
            self.level = 0
        
        if self.level == 1:
            self.mood = "snowy"
            self.slack_emoji = ":snow_cloud:"
        elif self.level == 2:
            self.mood = "rainy"
            self.slack_emoji = ":rain_cloud:"
        elif self.level == 3:
            self.mood = "cloudy"
            self.slack_emoji = ":cloud:"
        elif self.level == 4:
            self.mood = "fine"
            self.slack_emoji = ":mostly_sunny:" if self.temperature < 28 else ":sunny:"
        elif self.level == 5:
            self.mood = "sunny"
            self.slack_emoji = ":sunny:" if self.temperature < 28 else ":sunny: :sunny:"
        else:
            self.mood = "uncertain"
            self.slack_emoji = ":cyclone:"
            
    def intersection(self,feature_set):
        if len(set(self.description.split(' ')).intersection(feature_set))>0:
            return True
        else:
            return False
        