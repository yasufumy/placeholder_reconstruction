import requests


class LineNotifyApi:
    def __init__(self, access_token,
                 endpoint='https://notify-api.line.me/api/notify'):
        self.endpoint = endpoint
        self.headers = {
            'Authorization': 'Bearer {}'.format(access_token)
        }
        self.http_client = requests.Session()

    def post_message(self, messages):
        if not isinstance(messages, (list, tuple)):
            messages = [messages]
        data = {
            'message': '\n' + '\n'.join(messages)
        }
        self._post(data)

    def _post(self, data=None, timeout=None):
        url = self.endpoint
        headers = self.headers
        response = self.http_client.post(
                    url, headers=headers, data=data, timeout=timeout
                )
        return response


def load_access_token(filename):
    with open(filename) as f:
        access_token = f.read().strip()
    return access_token


if __name__ == '__main__':
    access_token = load_access_token('.linenotify')
    line_notify_api = LineNotifyApi(access_token)
    line_notify_api.post_message(['This is test.', 'This is also test.'])
