---
layout: post
title:  "Computer Science fundamental"
date:   2024-07-18 00:00:00 +0000
tags: Computer Science fundamental
color: rgb(200,200,200)
cover: '/assets/2024-06-01-cs/cs.png'
# Computer Science
---

Owner: Xiaofan SHEN
Tags: Infrastructure

# Python

### **Variables**

A **variable** in a program acts like a container. It can be used to store a string, a number, or other kinds of data.

![box-and-tangible-value-4-mol6CH.png](Computer%20Science%20d1ebabbae65e4cd39543506166c5471f/box-and-tangible-value-4-mol6CH.png)

**str1.find(str2)**

<aside>
💡 **str1.find(str2)**

</aside>

# Parallelism

- When multiple workers can split up a problem without adding in coordination overhead, computer scientists sometimes call the problem **embarrassingly parallel**.
- 

![Untitled](Computer%20Science%20d1ebabbae65e4cd39543506166c5471f/Untitled.png)

## Pipelining

**Pipelining** is a form of parallelism where tasks are solved faster by having multiple tasks simultaneously at various levels of completion.

Parallelism can improve performance by improving **latency** (the time that a task takes from beginning to end) or by improving **throughput** (the amount of tasks that can complete during a given interval of time).

# **Resource Tradeoffs**

Inexpensive computers can perform 3 basic computations in a nanosecond — that's one (wildly oversimplified) way of describing what it means to be a “3 GHz” computer.

## Caching

**Caching** is a general problem-solving approach that provides a much more flexible way of dealing with certain resource tradeoffs.

![**Hardware caches**](Computer%20Science%20d1ebabbae65e4cd39543506166c5471f/computer-processor-diagram-ULTlpu.png)

**Hardware caches**

## API

An API is an interface or menu that different programs can use to communicate with each other.

<aside>
💡 An **interface** is an abstraction that manages complexity by defining how to **interact** with a process while hiding how the process **actually** gets donLisLis

</aside>

*“向荆市长隐藏的信息允许档案部在不打扰荆市长的情况下改变其工作方式。这是界面意义的重要组成部分。”*

## Binary Search

**Bridges of Königsberg**

![konigsberg-bridge_-simplified-w-nsew-2-xpQnkF.png](Computer%20Science%20d1ebabbae65e4cd39543506166c5471f/konigsberg-bridge_-simplified-w-nsew-2-xpQnkF.png)

图中每条边都经过一次的路径称为**欧拉路径**。

<aside>
💡 一条路径只有一个端点和一个起点，因此如果一个图有两个以上的顶点，且连接边的数量为奇数，那么就不可能存在欧拉路径。

</aside>

<aside>
💡 移除柯尼斯堡的**任何一座**桥都只留下两个具有奇数条连接边的顶点。这意味着移除**任何**一座桥都可以形成欧拉路径。

</aside>

## **Breadth-first search  Depth-first search**

A **list** is a data type for storing multiple items of the same type that are related and belong together.

## List

![diff-variable-names-okay-2TIHqg.png](Computer%20Science%20d1ebabbae65e4cd39543506166c5471f/diff-variable-names-okay-2TIHqg.png)

## **pseudocode**

Writing the logic of programs in English is called **pseudocode**.

An **algorithm** is a step by step process designed to achieve some outcome, and computers are the fastest machines ever conceived for carrying out step by step processes!

That means that the study of algorithms is a core aim of computer science, but their use transcends any one discipline.

Code worked or delivered

When a computer scientist says an algorithm “works,” that doesn't just mean it promises to produce a correct result. It also has to deliver on that promise, *finish*, and produce the result, on every possible input.

Making sure an algorithm always finishes and produces a result is called *proving termination*, and it can be a little bit tricky!

 

Langton's ant、Boolean

## A **conditional statement** uses a Boolean expression to determine the code that is run.

Dictionary

![Untitled](Computer%20Science%20d1ebabbae65e4cd39543506166c5471f/Untitled%201.png)

```
cleaned_word = word.lower().strip(punctuation)
```

reader = open('data/jekyll.txt')
punctuation = '.;,-“’”:?—‘!()_'

freq_dict = {}
for line in reader:
for word in line.split():
cleaned_word = word.lower().strip(punctuation)
if cleaned_word in freq_dict:
freq_dict[cleaned_word]+=1
else:
freq_dict[cleaned_word]=1

print(len(freq_dict))

## **Counter**