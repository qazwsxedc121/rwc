import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { ElectronService } from 'app/core/services';
import { of } from 'rxjs';


class Tag {
  constructor(
    public name: string,
    public parent: string,
  ) {

  }
}
class Word {
  constructor(
    public name: string,
    public tag: string,
  ) {
  }
}

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

  constructor(
    private router: Router,
    private electronService: ElectronService
  ) {

  }
  ngOnInit(): void {

  }

  public newWord: string = "";
  public currentTag: string = "";
  public wordList: Array<Word> = [];
  public tagMap: Map<string, Tag> = new Map();
  public addWord(word: string, tag: string) {
    this.wordList.push(new Word(word, tag));
  }
  public addTag(tag: string, parent?: string) {
    if (this.tagMap.has(tag)) {
      return;
    }
    this.tagMap.set(tag, new Tag(tag, parent));
    this.tagList.push(tag);
  }
  public tagList: Array<string> = [];
  public keyEnter() {
    console.log(this.currentTag)
    this.addWord(this.newWord, this.currentTag);
    this.newWord = "";
  }
  public newTag: string = "";
  public tagKeyEnter() {
    this.addTag(this.newTag);
    this.newTag = ""
  }
  public save() {
    this.electronService.save(this.wordList);
  }
  public load() {
    let d = this.electronService.load();
    this.wordList = d;
  }

}
