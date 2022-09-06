import os,sys,io,json,random
from pathlib import Path
this_dir = os.path.split(os.path.realpath(__file__))[0]


'''
多热点，多模板多样性
'''

def get_filter_prompt():
    human_ref_data_dir = os.path.join(Path(this_dir).parent.parent,'eval_data','human_eval')
    filter_prompt_file_path = os.path.join(human_ref_data_dir,'data','meta_prompt.json')
    json_prompt = json.loads(io.open(filter_prompt_file_path,'r').read())['RECORDS']
    return json_prompt

word_ref_pool = {
    '#postive#':['高兴','欣慰','欢喜','爽','开心','舒服','心里美'],
    '#negtive#':['郁闷','难过','忧虑','难过','不好受','烦'],
    '#KEY#':['#KEY#'],
    '#QC#':['聊聊','说说','谈谈','讲讲','唠唠','掰扯掰扯','研究研究']
}


negtive_hotpoint = ['佩洛西窜访台湾','林志颖出车祸了','互联网大厂裁员','烂尾楼事件','研究所招聘硕士保安','我论文投NeurIPS被打了一个低分','薇娅偷税漏税','河南暴雨','新疆棉事件',
                    '新冠肺炎事件','瑞幸造假事件','西昌森林火灾','打卡式旅游','乔碧萝殿下','北大毕业生送外卖','大学生网贷','抖音式生活','超前消费','行车不规范 亲人两行泪',
                    '明星数据造假','娘炮文化','AI换脸','演员天价片酬','人造节日','骚扰电话','共享单车涨价','打击盗版电影','唐山打人案','我爸是李刚',
                    'P2P平台爆雷','高铁霸座男','美国枪击案','小镇做题家','网络炫富','出生率下降','程序员996加班']
postive_hotpoint = \
    ['新冠口服特效药','军工股涨停','网络直播带货','特斯拉造机器人','GPT3写论文研究自己','暗黑三手游上线','开放三胎','孟晚舟回国','鸿星尔克爆卖','东京奥运会',
     '神舟十二号火箭飞行任务成功','G20峰会将在沙特举行','5G网络将覆盖所有地级市','科教兴国','打工人才是人上人','国潮品牌','广场舞','垃圾分类','人工智能画家','文明旅游',
     '网络文学','正当防卫','中国锦鲤','建议专家不要建议']
all_hotpoint = negtive_hotpoint + postive_hotpoint

start_sentence_pool = [
    ['您好!','您好!'],
    ['人来的不少，我很#postive#。','八百年前的老词，又用到这了!'],
    ['各位同胞们!想死你们了!','特别是今晚第一次来到我们当中的，看谁谁顺眼，心里特#postive#。'],
    ['好险啊!','你这人有毛病，怎么一见面就大叫“好险啊”。'],
    ['亲爱的朋友们','接下来由我俩为大家演出'],
    ['您在这干嘛呢?','我们这是表演相声。'],

    ['朋友们好，新年吉祥!','给各位拜年了'],
    ['今天咱俩说什么?','今天咱俩#QC#点儿不一样的。'],
    ['在场的亲爱的观众朋友们!大家好!','今天晚上来人不少啊，满坑满谷!'],
    ['来了老弟？','嚯!,好久不见!'],
    ['我最近发现一个秘密！','给咱们大伙分享#QC#?']


]

under_take_sentence_pool = [
    ['最近在忙什么呀','说出来吓死你','什么呀这么神秘'],
    ['今天很#postive#呀','是，来了这么多的朋友'],
    ['今天我和我的搭档在这儿给大家讲一段','对，讲一段'],
    ['我最近有些#negtive#','嚯，有什么可#negtive#的呀'],
    ['今天特别#postive#','什么事儿呀这么#postive#'],
    ['今天特别#negtive#','什么事儿呀这么#negtive#'],
    ['别提了，#negtive#着呢!','什么事这么#negtive#?'],
]


inspired_sentence_pool = [
    ['今天我们来#QC##KEY#','#KEY#?'],
    ['我最近比较关心时事','那咱们来#QC##KEY#'],
    ['我最近比较关心时事','心系天下！','对，就比如这个#KEY#'],
    ['我最近比较关心时事','什么时事呀?','就比如最近的这个#KEY#'],
    ['最近有很多大事发生！','对，很多大事','就比如最近的#KEY#'],
    ['我们#QC#这个#KEY#'],
    ['你听说那个#KEY#了么','听说了，都在讨论这个'],
    ['你听说那个#KEY#了么','听说了，这可是个热点'],
    ['你听说那个#KEY#了么','怎么，您有研究？'],
    ['#KEY#,你没听说么!','听说了，都在讨论这个'],
    ['#KEY#,你没听说么!','听说了，这可是个热点'],
    ['#KEY#,你没听说么!','怎么，您有研究？'],
    ['我最近研究了这个#KEY#','嚯，您这厉害了','那可不是'],


]



if __name__ == '__main__':
    total_nums = 300
    start_candi_num = len(start_sentence_pool)
    under_candi_num = len(under_take_sentence_pool)
    inspired_candi_num = len(inspired_sentence_pool)
    prompt_res_list = set([])
    for idx in range(total_nums):
        start_sentence = start_sentence_pool[random.randint(0,start_candi_num-1)]
        undertake_sentence = under_take_sentence_pool[random.randint(0,under_candi_num-1)]
        inspired_sentence = inspired_sentence_pool[random.randint(0,inspired_candi_num-1)]
        hot_point = ''
        if 'negtive' in ''.join(undertake_sentence):
            hot_point = negtive_hotpoint[random.randint(0,len(negtive_hotpoint)-1)]
        elif 'postive' in ''.join(undertake_sentence):
            hot_point = postive_hotpoint[random.randint(0,len(postive_hotpoint)-1)]
        else:
            hot_point = all_hotpoint[random.randint(0,len(all_hotpoint)-1)]

        word_ref_candi = {k:word_ref_pool[k][random.randint(0,len(word_ref_pool[k]) - 1)] for k in word_ref_pool.keys()}
        prompt_list = start_sentence + undertake_sentence + inspired_sentence

        prompt_raw = '\n'.join([str(i % 2) + ':' + prompt_list[i] for i in range(len(prompt_list))])
        for key in word_ref_candi.keys():
            prompt_raw = prompt_raw.replace(key,word_ref_candi[key])
        format_prompt = prompt_raw.replace('#KEY#', hot_point)
        format_prompt = '#' * 3 + hot_point + '#' * 3 + '\n' + format_prompt
        prompt_res_list.add(format_prompt)
        # print()
    io.open(os.path.join(this_dir,'candi_fill_prompt.txt'),'w').write('\n\n'.join(prompt_res_list))
