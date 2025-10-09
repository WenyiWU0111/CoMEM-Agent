# MMInA: Benchmarking Multihop Multimodal Internet Agents Dataset

### Instruction:

The dataset consists $6$ folders.

- **normal:** 176 tasks. All of them are 2-hop or 3-hop tasks.

- **multi567:** 180 tasks. All 5-hop, 6-hop, 7-hop tasks are here.

- **compare:** 100 tasks. Some of 2-hop, 3-hop, 4-hop tasks are here. All tasks in this folder need to answer a comparable question first.

- **multipro:** 86 tasks. All 8-hop, 9-hop, 10-hop tasks are here.

- **shopping:** 200 tasks. All tasks here are about items in OneStopMarket

- **wikipedia:** 308 tasks. All tasks here are limited in wikipedia. Some are comparable tasks and others are simple. (108 tasks of them are filtered from [WebQA])

***"task_id"*** indicates the position of this task within the current folder.

***"start_url"*** is the webpage provided to the agent for initial access.

***"intent"*** and ***"intent_template"*** are the core of our tasks. The first part is telling agent the final state of each hop. The second part are some reference URLs to solve the task. The third part is our question.

***"procedure"*** refers to the evaluation method used in multi-hop tasks. (as mentioned in our paper)

For single-hop tasks, its evaluation method is reflected in ***'eval_types'***, and a reference answer is provided.

Here is an example of our task:

```json
{
    "sites": [
        "shopping"
    ],
    "task_id": 17,
    "require_login": true,
    "storage_state": "./.auth/shopping_state.json",
    "start_url": "https://library.kiwix.org/viewer#wikipedia_en_all_maxi_2024-01/A/User%3AThe_other_Kiwix_guy/Landing",
    "geolocation": null,
    "intent_template": "For actions 'book a hotel','book a car', 'book a flight','search on the Youtube', 'search on the twitter', 'search some events', 'Find food', 'Travel Guide', 'Exchange dollars': the action is finished just after click the search button! Attention: If you think all the actions had been done, return the final url as the answer!!! \n\n Here are some reference urls: Wiki: https://library.kiwix.org/viewer#wikipedia_en_all_maxi_2024-01/A/User%3AThe_other_Kiwix_guy/Landing  \nRent a car:https://sg.trip.com/carhire/?channelid=14409&locale=en-SG&curr=USD \nBook a flight:https://www.momondo.com/ \nBook a hotel:https://sg.trip.com/hotels/?locale=en-SG&curr=USD \nShopping:http://localhost:7770/ \nSearch an event:https://www.eventbrite.com/ \nTwitter:https://twitter.com/home \nYoutube:https://www.youtube.com/ \nFind food:https://www.timeout.com/ \nExchange dollars: https://www.xe.com/ \nTravel Guide:https://www.nomadicmatt.com \n\n\n Question:- Which city has a red tower, Tokyo or San Francisco? Help me check some events there.",
    "instantiation_dict": {},
    "intent": "For actions 'book a hotel','book a car', 'book a flight','search on the Youtube', 'search on the twitter', 'search some events', 'Find food', 'Travel Guide', 'Exchange dollars': the action is finished just after click the search button! Attention: If you think all the actions had been done, return the final url as the answer!!! \n\n Here are some reference urls: Wiki: https://library.kiwix.org/viewer#wikipedia_en_all_maxi_2024-01/A/User%3AThe_other_Kiwix_guy/Landing  \nRent a car:https://sg.trip.com/carhire/?channelid=14409&locale=en-SG&curr=USD \nBook a flight:https://www.momondo.com/ \nBook a hotel:https://sg.trip.com/hotels/?locale=en-SG&curr=USD \nShopping:http://localhost:7770/ \nSearch an event:https://www.eventbrite.com/ \nTwitter:https://twitter.com/home \nYoutube:https://www.youtube.com/ \nFind food:https://www.timeout.com/ \nExchange dollars: https://www.xe.com/ \nTravel Guide:https://www.nomadicmatt.com \n\n\n Question:- Which city has a red tower, Tokyo or San Francisco? Help me check some events there.",
    "require_reset": false,
    "eval": {
        "eval_types": [
            "string_match"
        ],
        "reference_answers": {
            "must_include": [
                "eventbrite",
                "tokyo"
            ]
        },
        "reference_url": "",
        "program_html": [],
        "string_note": "",
        "reference_answer_raw_annotation": []
    },
    "intent_template_id": 348,
    "cnt_hop": 3,
    "procedure": [
        "kiwix",
        "event",
        "end"
    ],
    "shop": "",
    "city": "tokyo",
    "flight": "tyo"
}
```


[WebQA]: https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.pdf