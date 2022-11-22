import argparse
import json
import requests
from mattermostdriver import Driver

def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--url",
            default='10.17.23.228',
            type=str
        )
    parser.add_argument(
            "--mm_port",
            default=8065,
            type=int
        )
    parser.add_argument(
            "--trans_port",
            default=14000,
            type=int
            )
    parser.add_argument(
            "--bot_access_token",
            default='1rrem7dyzty6znxoaapn8wmeic',
            type=str
            )
    parser.add_argument(
            "--login_id",
            default="kbbank@kbfg.com",
            type=str
            )
    parser.add_argument(
            "--password",
            default="kbbank123",
            type=str
            )

    args = parser.parse_args()

    return args

class TransBot():
    '''
    Allow to login mattermost channel
    '''
    def __init__(self, url, mm_port, trans_port, bot_access_token, login_id, password):

        self.base_url = f'http://{url}:{mm_port}'
        self.trans_endpoint = f'http://{url}:{trans_port}'
        self.bot_access_token = bot_access_token

        self.bot_id = requests.get(f'{self.base_url}/api/v4/users/me', headers={'Content-Type': 'application/json;charset=UTF-8', 'Authorization':f'Bearer {self.bot_access_token}'}).json()['id']

        self.log_info = requests.post(f'{self.base_url}/api/v4/users/login', headers={'Content-Type': 'application/json'}, data = json.dumps({"login_id":login_id,"password":password}))
        self.mm = Driver({
            'url': url,
            "token": self.log_info.headers.get('Token'), ## THE TOKEN THAT YOU GET FROM curl -i -d '{"login_id":"kbbank@kbfg.com","password":"kbbank123"}' http://10.17.23.228:8065/api/v4/users/login
            'scheme': 'http',
            'port': mm_port
        })
        self.mm.login()

    async def trans_handler(self, e):
        '''
        You can refer mattermost API V4 SPEC at https://documenter.getpostman.com/view/4508214/RW8FERUn#auth-info-bb2c27c2-5029-4448-9d56-46728343bac9
        '''
        message = json.loads(e)
        event = message.get('event', None)
        print('event', message)
        if event == 'posted':
            channel_id = message['broadcast']['channel_id']
            channel_name = message['data'].get('channel_name', None)
            src_lang, tgt_lang = {channel_name == 'enko': ('en_XX', 'ko_KR'), channel_name == 'koen' : ('ko_KR', 'en_XX')}.get(True, ('en_XX', 'ko_KR'))
            print(f"src lang : {src_lang}, tgt lang : {tgt_lang}")
            posted = message['data'].get("post", None)
            input_text = json.loads(posted).get("message", None)
            if json.loads(posted)['props'].get("from_bot", None) is None:
                if input_text == '':
                    pass

                elif input_text == "저장하기":
                    res = requests.get(f"{self.base_url}/api/v4/channels/{channel_id}/posts", headers={'Authorization': 'Bearer ' + self.log_info.headers.get('Token')})
                    sorted_posts = sorted(res.json()['posts'].items(), key = lambda item:item[1]['create_at'])
                    posts = [v['message'].replace('\n', ' ') for k,v in sorted_posts if v['message'] not in ['', '저장하기']]

                    file_path = f"./{src_lang}_to_{tgt_lang}.txt"
                    with open(file_path, "wb") as f:
                        f.write('\n'.join(posts).encode("utf-8"))

                    file_id = mm.files.upload_file(channel_id=channel_id, files={'files':(file_path, open(file_path, 'rb'))})['file_infos'][0]['id']
                    mm.posts.create_post(options={'channel_id':channel_id, 'file_ids':[file_id]})

                else:
                    trans = requests.post(self.trans_endpoint, data=json.dumps({'q':input_text, 'source':src_lang, 'target':tgt_lang})).json()['translatedText']
                    requests.post(url=f'{self.base_url}/api/v4/posts',
                            headers={'Content-Type': 'application/json;charset=UTF-8', 'Authorization':f'Bearer {self.bot_access_token}'},
                            data=json.dumps({"user_id" : self.bot_id, "channel_id":channel_id,
                                "message":f"**번역문**\n{trans}"}))

        else:
            pass

if __name__ == '__main__':
    args = define_argparser()
    transbot = TransBot(url=args.url, mm_port=args.mm_port, trans_port=args.trans_port, bot_access_token=args.bot_access_token,
            login_id=args.login_id, password=args.password)

    transbot.mm.init_websocket(transbot.trans_handler)
