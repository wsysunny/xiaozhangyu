import cherrypy
import API.detect
import API.search as sh

config = {
    'global' : {
        'server.socket_host' : '127.0.0.1',
        'server.socket_port' : 8080,
        'server.thread_pool' : 8,
        'server.max_request_body_size' : 0,
        'server.socket_timeout' : 60
  }
}

@cherrypy.expose
class API(object):
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self):
        return {"key": "value"}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def search(self, **kwargs):
        result = sh.search(**kwargs)
        return result
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def detect(self, img_data, img_type=1):
        img = Image()
        if img_type == '1':
            img.image_url = img_data
        if img_type == '2':
            img.image_file = img_data
        if img_type == '3':
            img.image_base64 = img_data
        else:
            pass  # TODO
        result = detect(img)
        return result

    

    
if __name__ == '__main__':
    cherrypy.quickstart(API(), '/', config)
