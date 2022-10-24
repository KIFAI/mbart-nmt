from mattermostdriver import Driver
import json
import requests


log_info = requests.post('http://10.17.23.228:8065/api/v4/users/login', headers={'Content-Type': 'application/json'}, data = json.dumps({"login_id":"kbbank@kbfg.com","password":"kbbank123"}))
mm = Driver({
    'url': '10.17.23.228',
    "token":log_info.headers.get('Token'), ## THE TOKEN THAT YOU GET FROM curl -i -d '{"login_id":"kbbank@kbfg.com","password":"kbbank123"}' http://10.17.23.228:8065/api/v4/users/login
    'scheme': 'http',
    'port': 8065
})
mm.login()

bot_access_token = '1rrem7dyzty6znxoaapn8wmeic'
bot_id = requests.get('http://10.17.23.228:8065/api/v4/users/me', headers={'Content-Type': 'application/json;charset=UTF-8', 'Authorization':f'Bearer {bot_access_token}'}).json()['id']

async def trans_handler(e):
    message = json.loads(e)
    event = message.get('event', None)
    print('event', message)
    if event == 'posted':
        channel_id = message['broadcast']['channel_id']
        channel_name = message['data'].get('channel_name', None)
        posted = message['data'].get("post", None)
        input_text = json.loads(posted).get("message", None)
        if json.loads(posted)['props'].get("from_bot", None) is None:
            src_lang, tgt_lang = {channel_name == 'enko': ('en_XX', 'ko_KR'), channel_name == 'koen' : ('ko_KR', 'en_XX')}.get(True, ('en_XX', 'ko_KR'))
            print(f"src lang : {src_lang}, tgt lang : {tgt_lang}")
            trans = requests.post('http://10.17.23.228:14000', data=json.dumps({'q':input_text, 'source':src_lang, 'target':tgt_lang})).json()['translatedText']
            requests.post(url='http://10.17.23.228:8065/api/v4/posts',
                    headers={'Content-Type': 'application/json;charset=UTF-8', 'Authorization':f'Bearer {bot_access_token}'},
                    data=json.dumps({"user_id" : bot_id, "channel_id":channel_id,
                        "message":f"**번역문 : {trans}"}))
            '''
            #Webhook interface
            requests.post(
                    url='http://10.17.23.228:8065/hooks/y5epyuk8xtfh7cfmafwfp6oh4h',
                    headers={'Content-Type':'application/json;charset=UTF-8', 'Authorization':f'Bearer {bot_access_token}'},
                    data=json.dumps(
                        {"username":"kbadmin", "channel":channel_name,
                            "attachments":[{"text":f"**번역문 : {trans}"}]}))
            '''
    else:
        pass

mm.init_websocket(trans_handler)
