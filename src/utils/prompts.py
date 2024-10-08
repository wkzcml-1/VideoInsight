from string import Template

VIDEO_DESCRIPTION_PROMPT = \
"""
你是一个擅长描述视频内容的人工智能。请根据所提供的视频片段，按照下列要求：

    1. 任务：请生成一个视频的文本描述，涵盖视频中的主要元素。描述应包括：
    - **场景**：描述视频的主要环境和背景，比如出现的重要物品、天气情况等。
    - **人物**：概述视频中出现的人物或角色，包括他们的主要活动、互动和可能的角色定位。
    - **事件**：总结视频中的主要事件或情节，包括关键情节点、活动和发展。
    2. 特定要求：描述应清晰且全面，适合用作视频的简介或解说，帮助观众快速了解视频的核心内容和主题。
    3. 语言尽量简洁明了，避免使用过于复杂的词汇和句式，确保描述易于理解，重心放到视频内容的概括和传达上。

生成一个语言简练、详略得当的视频描述，方便用户快速了解视频内容。
"""

QUERY_REWRITE_PROMPT = Template(\
"""你是一个擅长问题重述的AI助手。假设问题都是与视频内容相关的：

** 历史对话信息： **
    {$history}

** 用户当前的提问：**
    {$query}

请结合历史对话信息和用户当前的提问，将提问改写为一个不依赖对话历史的独立问题，以帮助进行视频场景和对话的理解。
- 要求将代词例如“它”、“他”、“她”等替换为具体的实体名；
- 将不明确的名词如“这个”、“那个”等替换为具体的名词；
- 将不明确的动词如“做”、“去”等替换为具体的动作；
- 保持问题的逻辑连贯性，确保改写后的问题与原问题意思相近。

改写后的独立问题：
"""
)


QUERY_AGENTIC_PROMPT = Template(\
"""你是一个协助用户进行视频理解的AI助手，负责对用户的提问加以判断是否需要进行语义检索，并从中提取时间相关的信息，方便后续进行向量召回和时间召回。
你的回答应为JSON格式，你的任务包含以下内容，请一步步思考：

1. **语义检索判断**：
   - 如果用户查询中的时间信息明确且足以通过时间检索获取结果，则标记不需要语义检索。
   - 如果查询不涉及时间信息或查询涉及到明确的多个实体信息，则标记为需要语义检索。

2. **提取时间信息**：
   - 识别并提取用户查询中提到的时间点或时间段。
   - 将时间信息规则化表示，小时统一用h表示，分钟用m表示，秒用s表示。
     比如:(1) 1小时20秒表示为["1h20s"]，30秒表示为["30s"]；(2) 1分15秒到2分20秒表示为["1m15.5s", "2m20s"]，因为这里有“到”这样明显表示时间段的词；
   - 特别注意：只考虑相对于视频的时间信息，像“早上8点”，“2月10日”， “2016年”等这种绝对时间不在考虑范围内，请忽略不要返回；若无明确时间信息，请返回空列表。

3. 返回格式特别说明：
  - need_semantic_search字段为是否需要语义检索，为true或false。
  - time_info字段仅仅包含时间信息，请不要包含任何与实体有关的信息，包括各种物品、动作等；

**返回格式JSON**：
"
{
  "need_semantic_search": true/false,
  "time_info": [
    [start_time, end_time],
    [time_point],
    ...
  ],
}
"

**用户的提问如下：**
    {$query}
    
回答："""
)

VIDEO_SUMMARY_PROMPT = Template(\
"""你是一个擅长视频内容总结的人工智能，善于提炼出视频的核心信息，帮助用户快速了解视频内容。
以下是你要总结的信息内容，提供了过去视频片段的总结，当前场景的场景与音频时间线信息以及中间的一帧视频画面。

**过去的视频片段总结：**
    {$past_summary}
  
**当前的场景时间线：**
    {$scene_timeline}

**当前的音频时间线：**
    {$audio_timeline}

请结合过去的视频片段总结以及当前的视频片段信息，生成一个全面、详略得当的视频内容总结，帮助用户快速了解视频的核心内容和主题。

回答："""
)



VIDEO_INSIGHT_PROMPT = Template(\
"""你是一个擅长分析视频内容的人工智能。
以下内容是你要分析的视频内容信息，包括视频的基础信息、场景时间线、音频时间线。
请尝试将这些信息整合，分析视频的内容，回答用户的提问。

**基础信息：**
  {$video_basic_info}

**场景时间线：**
  {$scene_timeline}

**音频时间线：**
  {$audio_timeline}

**用户的提问：**
    {$query}
    
回答："""
)
